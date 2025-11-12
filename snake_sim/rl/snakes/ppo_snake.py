import torch
import numpy as np
import threading
import time
from pathlib import Path

from snake_sim.rl.snakes.rl_snake_base import RLSnakeBase
from snake_sim.rl.types import PPOMetaData, PendingTransition, State
from snake_sim.rl.models.ppo_model import SnakePPONet
from snake_sim.rl.rl_data_queue import RLPendingTransitCache
from snake_sim.rl.constants import ACTION_ORDER_INVERSE
from snake_sim.rl.snapshot_manager import SnapshotManager


class PPOSnake(RLSnakeBase):
    """A snake controlled by a Proximal Policy Optimization (PPO) agent with optional snapshot hot-reload.

    Parameters:
        snapshot_dir: Directory where trainer saves snapshots (<base>_step*.pt). If provided, snake polls for newer weights.
        snapshot_base_name: Base prefix used by trainer (default 'ppo_model'). Used only for documentation; discovery is pattern-based.
        poll_interval: Seconds between polling attempts.
        auto_reload: Enable background polling thread when True.
        eager_first_load: Attempt immediate load of latest snapshot before first action if True.
    """

    def __init__(
        self,
        *,
        snapshot_dir: str | None = None,
        snapshot_base_name: str = 'ppo_model',
        poll_interval: float = 30.0,
        auto_reload: bool = True,  # Re-enabled for model weight updates
        eager_first_load: bool = True,
    ):
        super().__init__()
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._model: torch.nn.Module | None = None
        self._pending_cache = RLPendingTransitCache()
        self._transition_counter = 0
        # Snapshot manager for hot-reload
        self._snapshot_manager: SnapshotManager | None = None
        if snapshot_dir:
            def model_factory():
                return SnakePPONet(5, 9)  # Default: 5 channels, 9 context dims
            self._snapshot_manager = SnapshotManager(
                dir=snapshot_dir,
                base_name=snapshot_base_name,
                factory=model_factory
            )
        self._poll_interval = poll_interval
        self._auto_reload = auto_reload and (self._snapshot_manager is not None)
        self._eager_first_load = eager_first_load
        self._reload_thread: threading.Thread | None = None
        self._reload_running = False
        self._hot_reload_lock = threading.Lock()

    def _ensure_model(self, state: State):
        if self._model is not None:
            return
        ctx_dim = 0 if state.ctx is None else state.ctx.shape[0]
        in_channels = state.map.shape[0]
        self._model = SnakePPONet(in_channels, ctx_dim).to(self._device)
        
        # Update snapshot manager factory with correct dimensions
        if self._snapshot_manager:
            def model_factory():
                return SnakePPONet(in_channels, ctx_dim)
            self._snapshot_manager.factory = model_factory
            
            if self._eager_first_load:
                self._try_load_latest()
        if self._auto_reload and not self._reload_running:
            self._start_reload_thread()

    # ---- Snapshot reload helpers ----
    def _start_reload_thread(self):
        self._reload_running = True

        def _runner():
            while self._reload_running:
                try:
                    self._try_load_latest()
                except Exception:
                    pass  # suppress transient I/O or partial write issues
                time.sleep(self._poll_interval)

        self._reload_thread = threading.Thread(target=_runner, name=f"ppo_snake_reload_{self.get_id()}", daemon=True)
        self._reload_thread.start()

    def stop_reload(self):
        self._reload_running = False
        if self._reload_thread is not None:
            self._reload_thread.join(timeout=1.0)
            self._reload_thread = None

    def _try_load_latest(self) -> bool:
        if not self._snapshot_manager or not self._model:
            return False
        try:
            success = self._snapshot_manager.has_new_snapshot()
            if not success:
                return False
            with self._hot_reload_lock:
                return self._snapshot_manager.reload_into(self._model)
        except Exception:
            return False

    def _select_action(self, state: State):
        # Collate single state into expected tensor batch
        map_tensor = torch.from_numpy(state.map).unsqueeze(0).float().to(self._device)
        if state.ctx is not None:
            ctx_tensor = torch.from_numpy(state.ctx).unsqueeze(0).float().to(self._device)
            batch_in = {'map': map_tensor, 'ctx': ctx_tensor}
        else:
            batch_in = map_tensor
        with torch.no_grad():
            with self._hot_reload_lock:
                logits, value = self._model(batch_in)
        dist = torch.distributions.Categorical(logits=logits.squeeze(0))
        action_tensor = dist.sample()
        action_idx = int(action_tensor.item())
        log_prob = float(dist.log_prob(action_tensor).item())
        value_estimate = float(value.squeeze(0).item())
        return action_idx, log_prob, value_estimate

    def _next_step_for_state(self, state: State):
        # Initialize model lazily (ctx dimension known only after adapters applied)
        self._ensure_model(state)
        action_idx, log_prob, value_estimate = self._select_action(state)
        meta = PPOMetaData(log_prob=log_prob, value_estimate=value_estimate)
        pending = PendingTransition(
            state=state,
            action_index=action_idx,
            meta=meta,
            transition_nr=self._transition_counter,
            snake_id=self.get_id(),
        )
        self._pending_cache.add_transition(pending)
        self._transition_counter += 1
        # Translate action index -> direction Coord -> absolute next head coordinate
        direction = ACTION_ORDER_INVERSE[action_idx]
        next_coord = self._head_coord + direction
        return next_coord
