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
        mask_fatal_actions: bool = True,
        auto_reload: bool = True,
        eager_first_load: bool = True,
        deterministic: bool = True,  # Set True for deployment/evaluation
        fast_mode: bool = True,  # Enable inference speed optimizations
        use_half: bool = True,   # Convert model & inputs to FP16 if GPU available
    ):
        super().__init__()
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Runtime model instance (lazy init from first built state)
        self._model = None  # type: torch.nn.Module | None
        self._pending_cache = RLPendingTransitCache()
        self._transition_counter = 0
        self._mask_fatal_actions = mask_fatal_actions
        # Snapshot manager for hot-reload
        self._snapshot_manager: SnapshotManager | None = None
        # Snapshot manager factory will be replaced once first state encountered
        def model_creator():
            return model_factory(5, 0)  # Placeholder dims; overwritten on first ensure
        self._snapshot_manager = SnapshotManager(
            dir_name=snapshot_dir,
            base_name=snapshot_base_name,
            factory=model_creator
        )
        self._auto_reload = auto_reload and (self._snapshot_manager is not None)
        self._eager_first_load = eager_first_load
        self._deterministic = deterministic
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
        dtype = torch.float16 if (self._use_half and self._device.type == 'cuda') else torch.float32
        map_tensor = torch.from_numpy(state.map).unsqueeze(0).to(self._device, dtype=dtype)
        ctx_tensor = torch.from_numpy(state.ctx).unsqueeze(0).to(self._device, dtype=dtype)
        af_tensor = torch.from_numpy(state.meta['action_features']).unsqueeze(0).to(self._device, dtype=dtype)
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
        if self._mask_fatal_actions:
            action_mask = state.meta.get('action_mask')
            if action_mask is None:
                raise ValueError("action_mask missing in state.meta; ensure adapter ran")
            mask_tensor = torch.from_numpy(action_mask).to(logits.device)

            valid_count = int(mask_tensor.sum().item())
            masked_logits = logits.clone()
            if valid_count > 0:
                masked_logits = masked_logits.masked_fill(mask_tensor.unsqueeze(0) == 0, float('-inf'))
                dist = torch.distributions.Categorical(logits=masked_logits.squeeze(0))
            else:
                dist = torch.distributions.Categorical(logits=logits.squeeze(0))
        else:
            valid_count = -1  # masking disabled
            dist = torch.distributions.Categorical(logits=logits.squeeze(0))
        
        debug.debug_print(f"PPO Snake {self.get_id()} logits:", logits.cpu().numpy())
        debug.debug_print(f"PPO Snake {self.get_id()} action distribution probs:", dist.probs.cpu().numpy())
        if debug.is_debug_active():
            print("State context:", state.ctx)
            print("State meta:", state.meta)
            self._print_map()
        if self._deterministic:
            # Deterministic mode: pick best among VALID actions (masked_logits has -inf for invalid)
            target_logits = masked_logits if valid_count > 0 else logits
            action_tensor = target_logits.squeeze(0).argmax()
        else:
            # Stochastic mode: sample from distribution (for training/exploration)
            action_tensor = dist.sample()
        
        action_idx = int(action_tensor.item())
        log_prob = float(dist.log_prob(action_tensor).item())
        value_estimate = float(value.squeeze(0).item())
        if valid_count > 0 and mask_tensor[action_idx].item() == 0:
            if not hasattr(self, '_invalid_action_samples'):
                self._invalid_action_samples = 0
            self._invalid_action_samples += 1
            log.warning(
                f"Sampled invalid action index={action_idx} despite masking; performing safe fallback to best valid action."  # noqa: E501
            )
            safe_logits = masked_logits.squeeze(0)
            fallback_tensor = safe_logits.argmax()
            action_idx = int(fallback_tensor.item())
            # Recompute log_prob under distribution for fallback (prob ~0 if originally invalid)
            log_prob = float(dist.log_prob(fallback_tensor).item())
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
