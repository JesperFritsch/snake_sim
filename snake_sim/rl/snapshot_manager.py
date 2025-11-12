"""Unified SnapshotManager for model hot-reload between trainer and agents.

Features:
  - Discovers latest snapshot (highest step suffix) using model_snapshot helpers.
  - Atomic save (temp file + rename) to prevent partial reads.
  - Optional retention: prune older snapshots beyond a keep-last N.
  - Change detection: can check if a new snapshot appeared since last load.
  - Supports optimizer state persistence (optional) for trainer continuity.

Filename convention: <base_name>_step<STEP>.pt (same as model_snapshot).

Intended usage patterns:

Trainer side:
  sm = SnapshotManager(dir="models/ppo", base_name="ppo_model", factory=create_model)
  model = sm.init_or_load(device)
  ... training loop ...
  sm.save(step=global_update, model=model, policy_state=True, optimizers={'policy': opt1, 'value': opt2})
  sm.prune(keep_last=5)

Agent side (periodic polling):
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

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Callable
import time
import torch

from snake_sim.rl.model_snapshot import find_latest_snapshot, atomic_save


@dataclass
class LoadedSnapshot:
    path: Path
    step: int
    timestamp: float


class SnapshotManager:
    def __init__(self, dir: str | Path, base_name: str, factory: Callable[[], torch.nn.Module]):
        self.dir = Path(dir)
        self.base_name = base_name
        self.factory = factory
        self._last_loaded: Optional[LoadedSnapshot] = None

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
        data = torch.load(path, map_location=model.device if hasattr(model, 'device') else 'cpu')
        if isinstance(data, dict) and 'policy_state' in data:
            state_dict = data['policy_state']
        else:
            state_dict = data
        model.load_state_dict(state_dict)
        self._last_loaded = LoadedSnapshot(path=path, step=step, timestamp=time.time())

    # ---- Saving ----
    def save(self, step: int, model: torch.nn.Module, *, policy_state: bool = True, optimizers: Optional[Dict[str, torch.optim.Optimizer]] = None) -> Path:
        payload = {'policy_state': model.state_dict(), 'step': step}
        if optimizers:
            for name, opt in optimizers.items():
                payload[f'optimizer_{name}'] = opt.state_dict()
        return atomic_save(payload, self.dir, self.base_name, step=step)

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

    # ---- Retention ----
    def prune(self, keep_last: int = 5):
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
