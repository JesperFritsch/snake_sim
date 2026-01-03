"""Unified SnapshotManager for model hot-reload between trainer and agents.

Features:
  - Discovers latest snapshot (highest step suffix) using model_snapshot helpers.
  - Atomic save (temp file + rename) to prevent partial reads.
  - Optional retention: prune older snapshots beyond a keep-last N.
  - Change detection: can check if a new snapshot appeared since last load.
  - Supports optimizer state persistence (optional) for trainer continuity.
  - Watchdog integration: immediate notification when new snapshots are created (optional).

Filename convention: <base_name>_step<STEP>.pt (same as model_snapshot).

Intended usage patterns:

Trainer side:
  sm = SnapshotManager(dir="models/ppo", base_name="ppo_model", factory=create_model)
  model = sm.init_or_load(device)
  ... training loop ...
  sm.save(step=global_update, model=model, policy_state=True, optimizers={'policy': opt1, 'value': opt2})
    # Optional: call sm.prune(keep_last=K) if you explicitly want rotation.
    # This project typically relies on `archive/` for long-term retention instead.

Agent side (with watchdog - immediate reload):
  sm = SnapshotManager(dir="models/ppo", base_name="ppo_model", factory=create_model)
  model = sm.init_or_load(device)
  sm.start_watching(on_new_snapshot=lambda: sm.reload_into(model))

Agent side (periodic polling - legacy):
  sm = SnapshotManager(dir="models/ppo", base_name="ppo_model", factory=create_model)
  model = sm.init_or_load(device)
  while running:
      if sm.has_new_snapshot():
          sm.reload_into(model)
      sleep(poll_interval)

Note: Multi-process safety relies on atomic rename provided by underlying filesystem.
If running on NFS or exotic FS without atomic rename guarantees, additional locking may be needed.
"""
from __future__ import annotations

import logging

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Callable
import time
import torch
import threading
from importlib import resources

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent

from snake_sim.rl.model_snapshot import find_latest_snapshot, atomic_save

log = logging.getLogger(Path(__file__).stem)
logging.getLogger('watchdog').setLevel(logging.WARNING)
SNAPSHOT_BASE_DIR = Path(resources.files('snake_sim') / 'rl' / 'models_snapshots')

@dataclass
class LoadedSnapshot:
    path: Path
    step: int
    timestamp: float


class _SnapshotFileHandler(FileSystemEventHandler):
    """Watchdog handler for snapshot file changes."""
    
    def __init__(self, manager: 'SnapshotManager'):
        self.manager = manager
    
    def on_created(self, event: FileSystemEvent):
        """Called when a file is created."""
        if event.is_directory:
            return
        
        path = Path(event.src_path)
        # Check if this is a snapshot file (not a temp file)
        if path.suffix == '.pt' and '_step' in path.name and not path.name.startswith('.'):
            # Extract step to verify it's our snapshot
            step = self.manager._extract_step(path)
            if step is not None and path.name.startswith(self.manager.base_name):
                log.debug(f"📁 Watchdog detected new snapshot: {path.name} (step {step})")
                self.manager._notify_callbacks()
    
    def on_moved(self, event: FileSystemEvent):
        """Called when a file is moved (atomic rename completion)."""
        if event.is_directory:
            return
        
        dest_path = Path(event.dest_path)
        # Check if destination is a snapshot file
        if dest_path.suffix == '.pt' and '_step' in dest_path.name and not dest_path.name.startswith('.'):
            step = self.manager._extract_step(dest_path)
            if step is not None and dest_path.name.startswith(self.manager.base_name):
                log.debug(f"📁 Watchdog detected snapshot completion: {dest_path.name} (step {step})")
                self.manager._notify_callbacks()


class SnapshotManager:
    def __init__(
        self,
        dir_name: str | Path,
        base_name: str,
        factory: Callable[[], torch.nn.Module],
        *,
        archive_every_n: int = 2000,
        archive_subdir: str = "archive",
    ):
        self.dir = Path(SNAPSHOT_BASE_DIR, dir_name)
        self.base_name = base_name
        self.factory = factory
        self.archive_every_n = int(archive_every_n)
        self.archive_dir = self.dir / archive_subdir
        self._last_loaded: Optional[LoadedSnapshot] = None
        
        # Watchdog support
        self._observer = None
        self._callbacks: list[Callable[[], None]] = []
        self._callback_lock = threading.Lock()
        self._watching = False

    # ---- Loading ----
    def init_or_load(self, device: torch.device) -> torch.nn.Module:
        model = self.factory().to(device)
        latest = find_latest_snapshot(self.dir)
        if latest:
            self._load_path_into(latest.path, model, latest.step)
        return model

    def has_new_snapshot(self) -> bool:
        latest = find_latest_snapshot(self.dir)
        if not latest:
            return False
        if self._last_loaded is None:
            return True
        return latest.step > self._last_loaded.step

    def reload_into(self, model: torch.nn.Module) -> bool:
        latest = find_latest_snapshot(self.dir)
        if not latest:
            return False
        if self._last_loaded and latest.step == self._last_loaded.step:
            return False
        self._load_path_into(latest.path, model, latest.step)
        return True

    def _load_path_into(self, path: Path, model: torch.nn.Module, step: int):
        log.debug(f"Loading snapshot from {path} at step {step}")
        data = torch.load(path, map_location=model.device if hasattr(model, 'device') else 'cpu')
        if isinstance(data, dict) and 'policy_state' in data:
            state_dict = data['policy_state']
        else:
            state_dict = data

        # ---- Backwards compatibility: residual block refactor ----
        # Older snapshots used a ModuleList `res_blocks` with keys like:
        #   res_blocks.0.conv1.weight
        # Newer models inline these layers as:
        #   res1_conv1.weight
        # so loading would fail unless we remap.
        if (
            isinstance(state_dict, dict)
            and any(k.startswith('res_blocks.') for k in state_dict.keys())
            and not any(k.startswith('res1_conv1.') for k in state_dict.keys())
        ):
            remapped = dict(state_dict)
            for block_idx in (0, 1):
                for conv_idx in (1, 2):
                    for suffix in ('weight', 'bias'):
                        old_key = f"res_blocks.{block_idx}.conv{conv_idx}.{suffix}"
                        new_key = f"res{block_idx + 1}_conv{conv_idx}.{suffix}"
                        if old_key in remapped and new_key not in remapped:
                            remapped[new_key] = remapped.pop(old_key)
            state_dict = remapped

        model.load_state_dict(state_dict)

        # ---- Post-load numerical sanity check ----
        # A single NaN in parameters can make actor logits NaN and crash the whole run.
        # Refuse to mark snapshot as loaded if parameters are non-finite.
        try:
            for name, p in model.named_parameters():
                if p is None:
                    continue
                if not torch.isfinite(p).all():
                    raise ValueError(f"Non-finite parameter after load: {name}")
        except Exception as e:
            log.error(f"🚨 Refusing to load snapshot {path} at step {step}: {e}")
            return

        self._last_loaded = LoadedSnapshot(path=path, step=step, timestamp=time.time())

    # ---- Saving ----
    def save(
        self,
        step: int,
        model: torch.nn.Module,
        *,
        policy_state: bool = True,
        optimizers: Optional[Dict[str, torch.optim.Optimizer]] = None,
    ) -> Path:
        """Save a snapshot for hot-reload and optionally archive a copy every N steps.

        Behavior:
          - Always writes the latest snapshot to `self.dir` (so agents can load newest weights).
          - Additionally writes a copy to `self.archive_dir` when `step % archive_every_n == 0`.
            The archive directory is never pruned/rotated.

        Notes:
          - This method does not delete/rotate snapshots in `self.dir`.
            If you want rotation, call `prune(keep_last=...)` explicitly.
        """
        payload = {'policy_state': model.state_dict() if policy_state else model, 'step': step}
        if optimizers:
            for name, opt in optimizers.items():
                payload[f'optimizer_{name}'] = opt.state_dict()

        # Save latest (hot-reload target)
        latest_path = atomic_save(payload, self.dir, self.base_name, step=step)

        # Archive occasionally (no rotation)
        if self.archive_every_n > 0 and (step % self.archive_every_n == 0):
            try:
                atomic_save(payload, self.archive_dir, self.base_name, step=step)
            except Exception as e:
                log.warning(f"Failed saving archived snapshot at step {step}: {e}")

        return latest_path

    # ---- Information ----
    def get_latest_info(self) -> Optional[LoadedSnapshot]:
        """Get information about the latest available snapshot."""
        latest = find_latest_snapshot(self.dir)
        if latest:
            return LoadedSnapshot(path=latest.path, step=latest.step, timestamp=time.time())
        return None
    
    def get_last_loaded_info(self) -> Optional[LoadedSnapshot]:
        """Get information about the last successfully loaded snapshot."""
        return self._last_loaded

    # ---- Watchdog Integration ----
    def start_watching(self, on_new_snapshot: Optional[Callable[[], None]] = None) -> bool:
        """Start watching the snapshot directory for new files using watchdog.
        
        Args:
            on_new_snapshot: Optional callback to invoke when a new snapshot is detected.
                           If provided, it will be added to the callback list.
        
        Returns:
            True if watching started successfully, False if already watching.
        """
        if self._watching:
            log.debug("Watchdog already watching")
            if on_new_snapshot:
                self.add_callback(on_new_snapshot)
            return True
        
        # Ensure directory exists
        self.dir.mkdir(parents=True, exist_ok=True)
        
        # Create observer and handler
        self._observer = Observer()
        handler = _SnapshotFileHandler(self)
        self._observer.schedule(handler, str(self.dir), recursive=False)
        
        if on_new_snapshot:
            self.add_callback(on_new_snapshot)
        
        self._observer.start()
        self._watching = True
        log.info(f"👀 Watchdog started monitoring: {self.dir}")
        return True
    
    def stop_watching(self):
        """Stop the watchdog observer."""
        if not self._watching or self._observer is None:
            return
        
        self._observer.stop()
        self._observer.join(timeout=2.0)
        self._observer = None
        self._watching = False
        log.info("👀 Watchdog stopped")
    
    def add_callback(self, callback: Callable[[], None]):
        """Add a callback to be invoked when a new snapshot is detected.
        
        Args:
            callback: Function to call when new snapshot appears.
        """
        with self._callback_lock:
            if callback not in self._callbacks:
                self._callbacks.append(callback)
                log.debug(f"Added snapshot callback: {callback.__name__ if hasattr(callback, '__name__') else 'lambda'}")
    
    def remove_callback(self, callback: Callable[[], None]):
        """Remove a callback from the notification list.
        
        Args:
            callback: Function to remove from callbacks.
        """
        with self._callback_lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)
    
    def _notify_callbacks(self):
        """Notify all registered callbacks about a new snapshot."""
        with self._callback_lock:
            callbacks = self._callbacks.copy()
        
        for callback in callbacks:
            try:
                callback()
            except Exception as e:
                log.error(f"Error in snapshot callback: {e}", exc_info=True)

    # ---- Retention ----
    def prune(self, keep_last: int = 1):
        if keep_last <= 0:
            return
        snapshots = []
        latest = find_latest_snapshot(self.dir)
        # Collect all snapshots for ordering
        for p in self.dir.iterdir():
            if p.is_file() and p.suffix == '.pt':
                info = self._extract_step(p)
                if info is not None:
                    snapshots.append((p, info))
        snapshots.sort(key=lambda x: x[1], reverse=True)
        for p, _step in snapshots[keep_last:]:
            try:
                p.unlink()
            except Exception:
                pass

    def _extract_step(self, path: Path) -> Optional[int]:
        name = path.name
        if '_step' in name and name.endswith('.pt'):
            try:
                return int(name.split('_step')[-1].split('.pt')[0])
            except ValueError:
                return None
        return None
    
    # ---- Cleanup ----
    def __del__(self):
        """Cleanup watchdog observer on deletion."""
        self.stop_watching()
