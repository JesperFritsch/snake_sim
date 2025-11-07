
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Iterable, Tuple, Callable, Optional
import threading
import time
import math
import random

import numpy as np
import torch
import torch.nn.functional as F

from snake_sim.rl.rl_data_queue import RLMetaDataQueue
from snake_sim.rl.types import RLTransitionData, PPOMetaData

log = logging.getLogger(Path(__file__).stem)

rl_data_queue = RLMetaDataQueue()


@dataclass
class PPOTrainerConfig:
    gamma: float = 0.99
    lam: float = 0.95  # GAE lambda
    clip_range: float = 0.2
    policy_lr: float = 3e-4
    value_lr: float = 1e-3
    epochs: int = 4
    minibatch_size: int = 256
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    device: str = 'auto'
    snapshot_dir: Optional[str] = None  # directory of model snapshots
    snapshot_base_name: str = 'ppo_model'
    save_every_updates: int = 1  # save snapshot after this many successful updates


class PPOTrainer:
    """Minimal PPO trainer operating on RLTransitionData objects.

    Model contract (flexible):
      - model.forward(state_batch) -> (logits, value) OR a dict with keys 'logits','value'
      - model has separate optimizers: self.policy_optimizer, self.value_optimizer
        If not present we fall back to single self.optimizer updating all params.

    State collation: expects each transition.state_buffer to be either
      a numpy array or dictionary of arrays with matching first dimension after stacking.
    This is intentionally lightweight – adapt _collate_states for custom encodings.
    """

    def __init__(self, model_or_path=None, config: PPOTrainerConfig = None, model_factory: Optional[Callable[[], torch.nn.Module]] = None, snapshot_manager=None):
        """Initialize trainer.

        model_or_path: either a constructed model (nn.Module) OR a path (file or directory).
        If path provided, model_factory must be a callable returning an uninitialized model instance
        whose state_dict matches saved snapshots.
        """
        self._update_counter = 0
        self.cfg = config or PPOTrainerConfig()
        if self.cfg.device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(self.cfg.device)
        self.model: torch.nn.Module
        self._model_factory = model_factory
        self._snapshot_manager = snapshot_manager
        if snapshot_manager is not None:
            # SnapshotManager handles model creation and loading
            if model_factory is None:
                raise ValueError("model_factory must be provided when using snapshot_manager")
            self.model = snapshot_manager.init_or_load(self.device)
            self.cfg.snapshot_dir = str(snapshot_manager.dir)
        else:
            if isinstance(model_or_path, (str, Path)):
                if model_factory is None:
                    raise ValueError("model_factory must be provided when initializing from path")
                from snake_sim.rl.model_snapshot import find_latest_snapshot
                path = Path(model_or_path)
                if path.is_file():
                    self.model = model_factory().to(self.device)
                    self._load_snapshot(path)
                    self.cfg.snapshot_dir = str(path.parent)
                else:
                    self.model = model_factory().to(self.device)
                    latest = find_latest_snapshot(path)
                    if latest:
                        self._load_snapshot(latest.path)
                    self.cfg.snapshot_dir = str(path)
            else:
                if model_or_path is None:
                    if model_factory is None:
                        raise ValueError("Either model_or_path or model_factory must be provided")
                    self.model = model_factory().to(self.device)
                else:
                    self.model = model_or_path.to(self.device)
        # Create optimizers if model doesn't provide them
        if not hasattr(self.model, 'policy_optimizer') and not hasattr(self.model, 'optimizer'):
            self.model.policy_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.policy_lr)
        if not hasattr(self.model, 'value_optimizer') and not hasattr(self.model, 'optimizer'):
            self.model.value_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.value_lr)
        # Internal store: transition_id -> (advantage, return)
        self._adv_returns: Dict[str, Tuple[float, float]] = {}
        self._train_lock = threading.Lock()
        self._bg_thread: Optional[threading.Thread] = None
        self._bg_running = False
        # Persistent accumulation buffer to avoid losing transitions when below threshold
        log.info(f"PPOTrainer initialized device={self.device}")

    def train(self):
        """Run one PPO training cycle if enough data accumulated.

        Returns training stats dict or None if insufficient data.
        """
        with self._train_lock:
            # Add any newly ingested transitions to persistent buffer
            if rl_data_queue.size() < self.cfg.minibatch_size:
                return None  # Not enough yet – keep accumulating
            transitions = rl_data_queue.get_transitions()
            traj_map: Dict[Tuple[str, str], List[RLTransitionData]] = {}
            for t in transitions:
                snake_id = t.snake_id or 'snake'
                episode_id = t.episode_id or 'episode'
                traj_map.setdefault((snake_id, episode_id), []).append(t)
            for k in traj_map:
                traj_map[k].sort(key=lambda x: x.step_nr)
            all_trajs = list(traj_map.values())
            for traj in all_trajs:
                self._compute_gae(traj)
            processed = [t for traj in all_trajs for t in traj]
            batch = self._collate_batch(processed)
            stats = self._ppo_update(batch)
            # Clear buffer only after successful training step
            return stats

    # ---- Core Steps ----
    def _compute_gae(self, trajectory: List[RLTransitionData]):
        values = []
        rewards = []
        dones = []
        for t in trajectory:
            meta = t.meta
            if not isinstance(meta, PPOMetaData):
                raise ValueError("Transition meta must be PPOMetaData for PPO training")
            values.append(meta.value_estimate)
            rewards.append(t.reward)
            dones.append(float(t.done))
        values = np.array(values + [0.0], dtype=np.float32)  # bootstrap value (0 if terminal)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)
        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0
        for i in reversed(range(len(rewards))):
            mask = 1.0 - dones[i]
            delta = rewards[i] + self.cfg.gamma * values[i + 1] * mask - values[i]
            gae = delta + self.cfg.gamma * self.cfg.lam * mask * gae
            advantages[i] = gae
        returns = advantages + values[:-1]
        # Normalize advantages (per trajectory)
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std
        for t, adv, ret in zip(trajectory, advantages, returns):
            self._adv_returns[t.transition_id] = (float(adv), float(ret))

    def _collate_states(self, states: List):
        """Stack raw states (numpy arrays or torch tensors) into a batch tensor.

        Assumes each transition.state_buffer is exactly what the model expects per user contract.
        """
        s0 = states[0]
        if isinstance(s0, np.ndarray):
            return torch.from_numpy(np.stack(states)).float().to(self.device)
        if torch.is_tensor(s0):
            return torch.stack(states).to(self.device)
        raise TypeError(f"Unsupported state_buffer type: {type(s0)}; expected numpy array or torch Tensor")

    def _collate_batch(self, transitions: List[RLTransitionData]):
        states = [t.state_buffer for t in transitions]
        next_states = [t.next_state_buffer for t in transitions]
        actions = torch.tensor([t.action_index for t in transitions], dtype=torch.long, device=self.device)
        old_log_probs = torch.tensor([t.meta.log_prob for t in transitions], dtype=torch.float32, device=self.device)  # type: ignore
        advantages = torch.tensor([self._adv_returns[t.transition_id][0] for t in transitions], dtype=torch.float32, device=self.device)
        returns = torch.tensor([self._adv_returns[t.transition_id][1] for t in transitions], dtype=torch.float32, device=self.device)
        batch = {
            'states': self._collate_states(states),
            'next_states': self._collate_states(next_states),
            'actions': actions,
            'old_log_probs': old_log_probs,
            'advantages': advantages,
            'returns': returns
        }
        return batch

    def _forward(self, states):
        out = self.model(states)
        if isinstance(out, dict):
            logits = out['logits']
            values = out['value']
        else:
            logits, values = out
        return logits, values.squeeze(-1)

    def _ppo_update(self, batch: Dict):
        states = batch['states']
        actions = batch['actions']
        old_log_probs = batch['old_log_probs']
        advantages = batch['advantages']
        returns = batch['returns']

        logits, values = self._forward(states)
        dist = torch.distributions.Categorical(logits=logits)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        # Ratio
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_range, 1.0 + self.cfg.clip_range) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(values, returns)
        loss = policy_loss + self.cfg.value_coef * value_loss - self.cfg.entropy_coef * entropy

        # Optimize (single optimizer or split)
        if hasattr(self.model, 'optimizer'):
            self.model.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
            self.model.optimizer.step()
        else:
            # Policy + value separate
            self.model.policy_optimizer.zero_grad()
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
            self.model.policy_optimizer.step()
            self.model.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
            self.model.value_optimizer.step()

        stats = {
            'policy_loss': float(policy_loss.detach().cpu()),
            'value_loss': float(value_loss.detach().cpu()),
            'entropy': float(entropy.detach().cpu()),
            'batch_size': int(actions.shape[0])
        }
        log.info(f"PPO update: policy={stats['policy_loss']:.4f} value={stats['value_loss']:.4f} entropy={stats['entropy']:.4f} size={stats['batch_size']}")
        # Clear advantage/return store after use to prevent memory leak / stale reuse.
        self._adv_returns.clear()
        # Snapshot saving
        self._update_counter += 1
        if self._snapshot_manager and (self._update_counter % self.cfg.save_every_updates == 0):
            self._snapshot_manager.save(step=self._update_counter, model=self.model, optimizers={
                'policy': getattr(self.model, 'policy_optimizer', None),
                'value': getattr(self.model, 'value_optimizer', None)
            })
        elif self.cfg.snapshot_dir and (self._update_counter % self.cfg.save_every_updates == 0):
            self._save_snapshot()
        return stats

    # Future: implement minibatch epochs (currently single pass for simplicity)
    # This can be extended by slicing batch tensors and re-forwarding per epoch/minibatch.

    # ---- Background Thread API ----
    def start_background(self, interval_sec: float = 1.0):
        """Start a background thread that periodically invokes train().

        interval_sec: sleep duration between training attempts. If train() returns None
        (no data), a shorter backoff (interval_sec * 0.5) is used to be more responsive
        without busy-waiting.
        """
        if self._bg_running:
            log.warning("Background training already running")
            return
        self._bg_running = True

        def _runner():
            log.info("Background PPO training thread started")
            while self._bg_running:
                stats = self.train()
                # If we trained, maybe data flush; if not, poll faster until buffer fills
                sleep_time = interval_sec if stats else min(interval_sec * 0.5, 0.5)
                time.sleep(sleep_time)
            log.info("Background PPO training thread stopped")

        self._bg_thread = threading.Thread(target=_runner, name="ppo_trainer_bg", daemon=True)
        self._bg_thread.start()

    def stop_background(self, join: bool = True, timeout: Optional[float] = None):
        """Signal background thread to stop and optionally join it."""
        if not self._bg_running:
            return
        self._bg_running = False
        if join and self._bg_thread is not None:
            self._bg_thread.join(timeout=timeout)
            self._bg_thread = None

    def _load_snapshot(self, path: Path):
        try:
            data = torch.load(path, map_location=self.device)
            if isinstance(data, dict) and 'policy_state' in data:
                self.model.load_state_dict(data['policy_state'])
                log.info(f"Loaded snapshot {path}")
            else:
                # Assume entire state_dict
                self.model.load_state_dict(data)
                log.info(f"Loaded raw state_dict snapshot {path}")
        except Exception as e:
            log.warning(f"Failed loading snapshot {path}: {e}")

    def _save_snapshot(self):
        from snake_sim.rl.model_snapshot import atomic_save
        try:
            payload = {'policy_state': self.model.state_dict(), 'update': self._update_counter}
            path = atomic_save(payload, self.cfg.snapshot_dir, self.cfg.snapshot_base_name, step=self._update_counter)
            log.info(f"Saved snapshot {path}")
        except Exception as e:
            log.warning(f"Failed saving snapshot: {e}")

