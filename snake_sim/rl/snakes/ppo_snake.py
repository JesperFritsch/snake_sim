import torch
import numpy as np
import threading
import logging
from pathlib import Path

from snake_sim.rl.snakes.rl_snake_base import RLSnakeBase
from snake_sim.rl.types import PPOMetaData, PendingTransition, State
from snake_sim.rl.state_builder import print_state
from snake_sim.rl.models.ppo_model import model_factory
from snake_sim.rl.training.rl_data_queue import RLPendingTransitCache
from snake_sim.rl.constants import ACTION_ORDER_INVERSE
from snake_sim.rl.snapshot_manager import SnapshotManager
from snake_sim.rl.action_masking import apply_action_mask_to_logits
import snake_sim.debugging as debug 

# debug.activate_debug()
# debug.enable_debug_for_all()

log = logging.getLogger(Path(__file__).stem)

class PPOSnake(RLSnakeBase):
    """A snake controlled by a Proximal Policy Optimization (PPO) agent with watchdog-based hot-reload.

    Parameters:
        snapshot_dir: Directory where trainer saves snapshots (<base>_step*.pt). Watchdog monitors for instant reload.
        snapshot_base_name: Base prefix used by trainer (default 'ppo_model').
        auto_reload: Enable watchdog monitoring for instant model reload when True.
        eager_first_load: Attempt immediate load of latest snapshot before first action if True.
        deterministic: If True, use argmax (best action). If False, sample from distribution (exploration).
        fast_mode: Enable inference optimizations (channels_last, TorchScript) on CUDA.
        use_half: Convert model and inputs to FP16 for faster inference on CUDA.
    """

    def __init__(
        self,
        *,
        snapshot_dir: str,
        snapshot_base_name: str = 'ppo_model',
        auto_reload: bool = True,
        eager_first_load: bool = True,
        deterministic: bool = True,  # Set True for deployment/evaluation
        deterministic_temperature: float = 0.0,
        fast_mode: bool = True,  # Enable inference speed optimizations
        use_half: bool = True,   # Convert model & inputs to FP16 if GPU available
    ):
        super().__init__()
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Runtime model instance (lazy init from first built state)
        self._model = None  # type: torch.nn.Module | None
        self._pending_cache = RLPendingTransitCache()
        self._transition_counter = 0
        self._snapshot_dir = snapshot_dir
        # Snapshot manager for hot-reload
        self._snapshot_manager: SnapshotManager | None = None
        # Snapshot manager factory will be replaced once first state encountered
        def model_creator():
            # Placeholder dims; overwritten on first ensure. Keep in sync with the default
            # channel layout (incl. food_dist) to avoid surprises if a snapshot is loaded
            # before the first state is seen.
            return model_factory(6, 0)
        self._snapshot_manager = SnapshotManager(
            dir_name=self._snapshot_dir,
            base_name=snapshot_base_name,
            factory=model_creator
        )
        self._auto_reload = auto_reload and (self._snapshot_manager is not None)
        self._eager_first_load = eager_first_load
        self._deterministic = deterministic
        # If > 0, deterministic mode becomes "low-temperature sampling" instead of hard argmax.
        # This reduces brittleness when the top-1 and top-2 logits are close.
        self._deterministic_temperature = float(deterministic_temperature)
        self._hot_reload_lock = threading.Lock()
        self._fast_mode = fast_mode and (self._device.type == 'cuda')
        self._use_half = use_half and (self._device.type == 'cuda')
        self._scripted = False

    def _ensure_model(self, state: State):
        if self._model is not None:
            return
        if state.ctx is None:
            raise ValueError("State ctx missing; CompleteStateBuilder should provide ctx")
        if 'action_features' not in (state.meta or {}):
            raise ValueError("State meta missing action_features; adapters must supply them")
        if 'action_mask' not in (state.meta or {}):
            raise ValueError("State meta missing action_mask; adapters must supply it")
        ctx_dim = state.ctx.shape[0]
        in_channels = state.map.shape[0]
        self._model = model_factory(in_channels, ctx_dim).to(self._device)
        if self._fast_mode:
            try:
                self._model = self._model.to(memory_format=torch.channels_last)
            except Exception:
                pass
        if self._use_half and self._device.type == 'cuda':
            try:
                self._model = self._model.half()
            except Exception:
                pass
        self._model.eval()
        # TorchScript trace with full input dict (map, ctx, action_features)
        if self._fast_mode and not self._scripted:
            try:
                example_map = torch.from_numpy(state.map).unsqueeze(0).float().to(self._device)
                example_ctx = torch.from_numpy(state.ctx).unsqueeze(0).float().to(self._device)
                example_af = torch.from_numpy(state.meta['action_features']).unsqueeze(0).float().to(self._device)
                if self._use_half and self._device.type == 'cuda':
                    example_map = example_map.half(); example_ctx = example_ctx.half(); example_af = example_af.half()
                # NOTE: model doesn't consume action_mask; it's included so the traced signature matches
                # the real inference input dict if you choose to pass it through later.
                example_in = {'map': example_map, 'ctx': example_ctx, 'action_features': example_af}
                self._model = torch.jit.trace(self._model, example_in, strict=False)
                self._scripted = True
                log.info("PPOSnake: scripted inference model active")
            except Exception as e:
                log.warning(f"TorchScript trace failed; continuing without scripting: {e}")
        if self._snapshot_manager:
            def model_creator():
                return model_factory(in_channels, ctx_dim)
            self._snapshot_manager.factory = model_creator
            if self._eager_first_load:
                self._try_load_latest()
            if self._auto_reload:
                self._snapshot_manager.start_watching(on_new_snapshot=self._on_snapshot_update)
                log.info(f"🔄 PPOSnake {self.get_id()}: Watchdog monitoring for instant model reload")

    # ---- Snapshot reload helpers ----
    def stop_reload(self):
        """Stop watchdog monitoring."""
        if self._snapshot_manager:
            self._snapshot_manager.stop_watching()
    
    def _on_snapshot_update(self):
        """Callback invoked by watchdog when a new snapshot is detected."""
        if not self._model:
            return
        
        try:
            with self._hot_reload_lock:
                result = self._snapshot_manager.reload_into(self._model)
                if result:
                    self._model.eval()
                    if self._snapshot_manager._last_loaded:
                        step = self._snapshot_manager._last_loaded.step
                        log.debug(f"⚡ PPOSnake {self.get_id()}: Hot-reloaded weights at step {step}")
        except Exception as e:
            log.warning(f"Failed to reload snapshot in callback: {e}")

    def _try_load_latest(self) -> bool:
        if not self._snapshot_manager or not self._model:
            return False
        try:
            success = self._snapshot_manager.has_new_snapshot()
            if not success:
                return False
            with self._hot_reload_lock:
                result = self._snapshot_manager.reload_into(self._model)
                if result:
                    # Ensure model is in eval mode after loading new weights
                    self._model.eval()
                return result
        except Exception:
            return False

    def _select_action(self, state: State):
        if state.ctx is None:
            raise ValueError("State ctx required for PPO inference")
        if 'action_features' not in (state.meta or {}):
            raise ValueError("Missing action_features in state.meta")
        if 'action_mask' not in (state.meta or {}):
            raise ValueError("Missing action_mask in state.meta")
        dtype = torch.float16 if (self._use_half and self._device.type == 'cuda') else torch.float32
        # Ensure arrays are contiguous (torch.from_numpy doesn't accept negative strides)
        map_np = np.ascontiguousarray(state.map)
        ctx_np = np.ascontiguousarray(state.ctx)
        af_np = np.ascontiguousarray(state.meta['action_features'])
        am_np = np.ascontiguousarray(state.meta['action_mask'])
        map_tensor = torch.from_numpy(map_np).unsqueeze(0).to(self._device, dtype=dtype)
        ctx_tensor = torch.from_numpy(ctx_np).unsqueeze(0).to(self._device, dtype=dtype)
        af_tensor = torch.from_numpy(af_np).unsqueeze(0).to(self._device, dtype=dtype)
        action_mask = torch.from_numpy(am_np).unsqueeze(0).to(self._device)
        action_mask = action_mask.to(dtype=torch.bool)
        if self._fast_mode:
            try:
                map_tensor = map_tensor.to(memory_format=torch.channels_last)
            except Exception:
                pass
        batch_in = {'map': map_tensor, 'ctx': ctx_tensor, 'action_features': af_tensor}
        with torch.no_grad():
            with self._hot_reload_lock:
                logits, value = self._model(batch_in)
        logits = logits.float()  # stable distribution math

        # ---- Safety against numerical issues ----
        # If logits contain NaNs/Infs (can happen if the learner diverged and published a bad snapshot),
        # never crash the actor process. Fall back to a uniform distribution over valid actions.
        if not torch.isfinite(logits).all():
            # Prefer logging over throwing: crashing an actor kills the whole multi-env run.
            try:
                log.warning(
                    "PPO Snake %s produced non-finite logits; falling back to safe sampling.",
                    self.get_id(),
                )
            except Exception:
                pass
            valid = action_mask.squeeze(0).to(dtype=torch.bool)
            if valid.any():
                valid_idx = torch.nonzero(valid, as_tuple=False).squeeze(-1)
                action_tensor = valid_idx[torch.randint(len(valid_idx), (1,), device=self._device)][0]
                action_idx = int(action_tensor.item())
                # It's not meaningful to report a log_prob/value here; keep them finite.
                return action_idx, 0.0, 0.0
            # Absolute last resort: pick action 0.
            return 0, 0.0, 0.0

        logits = apply_action_mask_to_logits(logits, action_mask)

        # Base distribution (used for log_prob reporting in non-deterministic mode)
        dist = torch.distributions.Categorical(logits=logits.squeeze(0))
        
        debug.debug_print(f"PPO Snake {self.get_id()} logits:", logits.cpu().numpy())
        debug.debug_print(f"PPO Snake {self.get_id()} action distribution probs:", dist.probs.cpu().numpy())
        if debug.is_debug_active():
            print("State context:", state.ctx)
            print("State meta:", state.meta)
            self._print_map()
        if self._deterministic:
            if self._deterministic_temperature and self._deterministic_temperature > 0.0:
                # Less-brittle "deterministic": sample with low temperature.
                # T -> 0 approaches argmax; small T keeps occasional alternative when top-2 are close.
                t = max(self._deterministic_temperature, 1e-6)
                temp_logits = (logits.squeeze(0) / t)
                temp_dist = torch.distributions.Categorical(logits=temp_logits)
                action_tensor = temp_dist.sample()
                log_prob = float(temp_dist.log_prob(action_tensor).item())
            else:
                # Deterministic mode: pick best action from logits
                action_tensor = logits.squeeze(0).argmax()
                log_prob = float(dist.log_prob(action_tensor).item())
        else:
            # Stochastic mode: sample from distribution (for training/exploration)
            action_tensor = dist.sample()
            log_prob = float(dist.log_prob(action_tensor).item())
        
        action_idx = int(action_tensor.item())
        value_estimate = float(value.squeeze(0).item())
        # No mask fallback needed
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
