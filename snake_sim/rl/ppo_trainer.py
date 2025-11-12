
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import threading
import time
# (removed unused imports math, random)

import numpy as np
import torch
import torch.nn.functional as F

from snake_sim.rl.rl_data_queue import RLMetaDataQueue
from snake_sim.rl.types import RLTransitionData, PPOMetaData, State
from snake_sim.rl.snapshot_manager import SnapshotManager

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
    save_every_updates: int = 10  # save snapshot after this many successful updates


class PPOTrainer:
    """Minimal PPO trainer that creates its own model lazily from first batch of data."""

    def __init__(self, config: PPOTrainerConfig = None, snapshot_dir: str = None):
        """Initialize trainer. Model is created lazily from first batch."""
        self._update_counter = 0
        self.cfg = config or PPOTrainerConfig()
        if self.cfg.device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(self.cfg.device)
        
        self.model = None
        self.policy_optimizer = None
        self.value_optimizer = None
        self._snapshot_dir = snapshot_dir
        self._snapshot_manager = None
        self._adv_returns = {}
        self._train_lock = threading.Lock()
        self._bg_thread = None
        self._bg_running = False
        
        # Training metrics tracking
        self._metrics_history = {
            'policy_loss': [],
            'value_loss': [], 
            'entropy': [],
            'returns_mean': [],
            'advantages_mean': [],
            'batch_size': []
        }
        self._window_size = 20  # Rolling average window
        
        log.info(f"PPOTrainer initialized device={self.device}")

    def _ensure_model(self, first_state: State):
        """Create model lazily from first state to determine input dimensions."""
        if self.model is not None:
            return
        
        from snake_sim.rl.models.ppo_model import SnakePPONet
        ctx_dim = 0 if first_state.ctx is None else first_state.ctx.shape[0]
        in_channels = first_state.map.shape[0]
        self.model = SnakePPONet(in_channels, ctx_dim).to(self.device)
        
        # Create optimizers
        self.policy_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.policy_lr)
        self.value_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.value_lr)
        
        # Setup snapshot manager if directory provided
        if self._snapshot_dir:
            def model_factory():
                return SnakePPONet(in_channels, ctx_dim)
            self._snapshot_manager = SnapshotManager(
                dir=self._snapshot_dir,
                base_name=self.cfg.snapshot_base_name,
                factory=model_factory
            )
            # Try to load latest snapshot
            self._try_load_latest_snapshot()
        
        log.info(f"PPOTrainer created model: {in_channels} channels, {ctx_dim} context dims")

    def _update_metrics_history(self, stats: dict):
        """Update training metrics history and compute running averages."""
        for key, value in stats.items():
            if key in self._metrics_history:
                self._metrics_history[key].append(value)
                # Keep only recent history (strict limit to prevent memory growth)
                max_history = self._window_size * 3  # Allow some buffer but cap growth
                if len(self._metrics_history[key]) > max_history:
                    self._metrics_history[key] = self._metrics_history[key][-self._window_size:]

    def _compute_running_averages(self) -> dict:
        """Compute running averages for recent training metrics."""
        averages = {}
        for key, values in self._metrics_history.items():
            if values:
                recent_values = values[-self._window_size:]
                averages[f'{key}_avg'] = sum(recent_values) / len(recent_values)
                if len(values) >= 2:
                    # Compute trend (recent vs older values)
                    mid = len(values) // 2
                    older_avg = sum(values[:mid]) / mid if mid > 0 else values[0]
                    recent_avg = sum(values[mid:]) / (len(values) - mid)
                    trend = recent_avg - older_avg
                    averages[f'{key}_trend'] = trend
        return averages

    def _log_detailed_metrics(self, stats: dict):
        """Log detailed training metrics with trends and running averages."""
        self._update_metrics_history(stats)
        
        if self._update_counter % 5 == 0:  # Every 5 updates
            averages = self._compute_running_averages()
            
            # Performance indicators
            policy_loss_avg = averages.get('policy_loss_avg', 0)
            value_loss_avg = averages.get('value_loss_avg', 0) 
            entropy_avg = averages.get('entropy_avg', 0)
            returns_avg = averages.get('returns_mean_avg', 0)
            
            # Trends
            policy_trend = averages.get('policy_loss_trend', 0)
            value_trend = averages.get('value_loss_trend', 0)
            returns_trend = averages.get('returns_mean_trend', 0)
            
            # Performance assessment
            policy_direction = "↓" if policy_trend < -0.001 else "↑" if policy_trend > 0.001 else "→"
            value_direction = "↓" if value_trend < -0.001 else "↑" if value_trend > 0.001 else "→"
            returns_direction = "↑" if returns_trend > 0.001 else "↓" if returns_trend < -0.001 else "→"
            
            log.info(f"Training Metrics (Update {self._update_counter}):")
            log.info(f"  Policy Loss: {policy_loss_avg:.4f} {policy_direction} | Value Loss: {value_loss_avg:.4f} {value_direction}")
            log.info(f"  Entropy: {entropy_avg:.4f} | Returns: {returns_avg:.4f} {returns_direction}")
            
            # Learning indicators
            if returns_trend > 0.01:
                log.info("  🚀 Snakes showing improvement in returns!")
            elif returns_trend < -0.01:
                log.info("  📉 Returns declining - possible overfitting or poor exploration")
            
            if entropy_avg < 0.5:
                log.info("  ⚠️  Low entropy - snakes may be too deterministic")
            elif entropy_avg > 1.5:
                log.info("  🎲 High entropy - snakes exploring actively")

    def _try_load_latest_snapshot(self):
        """Try to load latest snapshot using SnapshotManager."""
        if not self._snapshot_manager:
            return
        try:
            success = self._snapshot_manager.reload_into(self.model)
            if success and self._snapshot_manager._last_loaded:
                loaded_info = self._snapshot_manager._last_loaded
                self._update_counter = loaded_info.step
                log.info(f"📁 Loaded snapshot from {loaded_info.path}, resuming from step {self._update_counter}")
                # Also load optimizer states if available
                data = torch.load(loaded_info.path, map_location=self.device)
                if isinstance(data, dict):
                    if 'optimizer_policy' in data and self.policy_optimizer:
                        self.policy_optimizer.load_state_dict(data['optimizer_policy'])
                    if 'optimizer_value' in data and self.value_optimizer:
                        self.value_optimizer.load_state_dict(data['optimizer_value'])
                    log.info("📈 Restored optimizer states")
        except Exception as e:
            log.warning(f"Failed to load snapshot: {e}")

    def get_queue_size(self):
        """Get current transition queue size."""
        return rl_data_queue.size()

    def get_update_count(self):
        """Get current number of training updates performed."""
        return self._update_counter

    def train(self):
        """Run one PPO training cycle if enough data accumulated.

        Returns training stats dict or None if insufficient data.
        """
        with self._train_lock:
            if rl_data_queue.size() < self.cfg.minibatch_size:
                return None  # Not enough yet – keep accumulating
            transitions = rl_data_queue.get_transitions()
            if not transitions:
                return None
            
            # Initialize model lazily from first transition
            self._ensure_model(transitions[0].state)
            
            traj_map: Dict[Tuple[str, str], List[RLTransitionData]] = {}
            for t in transitions:
                snake_id = t.snake_id or 'snake'
                episode_id = t.episode_id or 'episode'
                traj_map.setdefault((snake_id, episode_id), []).append(t)
            for k in traj_map:
                traj_map[k].sort(key=lambda x: x.transition_nr)
            all_trajs = list(traj_map.values())
            for traj in all_trajs:
                self._compute_gae(traj)
            processed = [t for traj in all_trajs for t in traj]
            batch = self._collate_batch(processed)
            stats = self._ppo_update(batch)
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

    # (legacy _collate_states removed)

    def _collate_batch(self, transitions: List[RLTransitionData]):
        # Use new State container; next_state not used for PPO update currently (GAE uses stored value_estimates)
        states = [t.state for t in transitions]
        actions = torch.tensor([t.action_index for t in transitions], dtype=torch.long, device=self.device)
        old_log_probs = torch.tensor([t.meta.log_prob for t in transitions], dtype=torch.float32, device=self.device)  # type: ignore
        advantages = torch.tensor([self._adv_returns[t.transition_id][0] for t in transitions], dtype=torch.float32, device=self.device)
        returns = torch.tensor([self._adv_returns[t.transition_id][1] for t in transitions], dtype=torch.float32, device=self.device)
        state_batch = self._collate_states(states)
        batch = {
            'states': state_batch,
            'actions': actions,
            'old_log_probs': old_log_probs,
            'advantages': advantages,
            'returns': returns
        }
        return batch

    def _collate_states(self, states: List[State]):
        """Collate a list of State objects into batched torch tensors.

        Returns torch.Tensor (B,C,H,W) if no ctx present; otherwise dict {'map': map_batch, 'ctx': ctx_batch}.
        Mixed presence of ctx is not allowed.
        """
        maps_np = [s.map for s in states]
        map_batch = torch.from_numpy(np.stack(maps_np)).float().to(self.device)
        ctx_all = [s.ctx for s in states]
        any_ctx = any(c is not None for c in ctx_all)
        all_ctx = all(c is not None for c in ctx_all)
        if any_ctx and not all_ctx:
            raise ValueError("Mixed presence of ctx in batch; ensure all or none have context")
        if not any_ctx:
            return map_batch
        # consistent dims
        k = ctx_all[0].shape[0]
        for c in ctx_all:
            if c.shape[0] != k:
                raise ValueError("Inconsistent ctx dimensions")
        ctx_batch = torch.from_numpy(np.stack(ctx_all)).float().to(self.device)
        return {'map': map_batch, 'ctx': ctx_batch}
    # ---- Forward & Update ----
    def _forward(self, state_batch):
        out = self.model(state_batch)
        if isinstance(out, dict):
            logits = out['logits']
            value = out['value']
        else:
            logits, value = out
        return logits, value.squeeze(-1)

    def _ppo_update(self, batch: Dict):
        states = batch['states']
        actions = batch['actions']
        old_log_probs = batch['old_log_probs']
        advantages = batch['advantages']
        returns = batch['returns']

        logits, values = self._forward(states)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.cfg.clip_range, 1 + self.cfg.clip_range) * advantages
        policy_loss = -torch.mean(torch.min(surr1, surr2))
        value_loss = F.mse_loss(values, returns)
        
        # Combined loss update to avoid retain_graph issues
        total_loss = policy_loss + self.cfg.value_coef * value_loss - self.cfg.entropy_coef * entropy
        
        try:
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
            self.policy_optimizer.step()
            self.value_optimizer.step()
        except RuntimeError as e:
            if "inplace operation" in str(e):
                log.warning(f"Gradient computation failed due to inplace operation, skipping update: {e}")
                return None
            else:
                raise

        self._update_counter += 1
        # Save snapshot periodically
        if self._snapshot_manager and (self._update_counter % self.cfg.save_every_updates == 0):
            self._save_snapshot()

        stats = {
            'policy_loss': float(policy_loss.item()),
            'value_loss': float(value_loss.item()),
            'entropy': float(entropy.item()),
            'update': self._update_counter,
            'advantages_mean': float(advantages.mean().item()),
            'returns_mean': float(returns.mean().item()),
            'batch_size': len(actions)
        }
        
        # Log detailed metrics with trends
        self._log_detailed_metrics(stats)
        
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
            last_stats_time = time.time()
            stats_interval = 10.0  # Print stats every 10 seconds
            
            while self._bg_running:
                try:
                    stats = self.train()
                except Exception as e:
                    log.error(f"Training error: {e}", exc_info=True)
                    stats = None
                
                # Log brief stats when we get them (every update)
                if stats:
                    try:
                        current_time = time.time()
                        
                        # Brief update log
                        log.info(f"Update {stats['update']}: "
                               f"P_Loss={stats['policy_loss']:.3f}, "
                               f"V_Loss={stats['value_loss']:.3f}, "
                               f"Return={stats['returns_mean']:.3f}, "
                               f"Batch={stats['batch_size']}")
                        
                        # Detailed stats less frequently 
                        if current_time - last_stats_time >= stats_interval:
                            queue_size = rl_data_queue.size()
                            log.info(f"📊 Training Overview (Update {stats['update']}):")
                            log.info(f"   Queue Size: {queue_size} | Batch Size: {stats['batch_size']}")
                            log.info(f"   Policy Loss: {stats['policy_loss']:.4f} | Value Loss: {stats['value_loss']:.4f}")
                            log.info(f"   Entropy: {stats['entropy']:.4f} | Avg Return: {stats['returns_mean']:.4f}")
                            last_stats_time = current_time
                    except Exception as e:
                        log.error(f"Logging error: {e}")
                
                # If we trained, maybe data flush; if not, poll faster until buffer fills
                sleep_time = interval_sec if stats else min(interval_sec * 0.5, 0.5)
                try:
                    time.sleep(sleep_time)
                except Exception:
                    break  # Exit gracefully on interruption
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

    def _save_snapshot(self):
        """Save model snapshot using SnapshotManager."""
        if not self._snapshot_manager:
            return
        try:
            optimizers = {
                'policy': self.policy_optimizer,
                'value': self.value_optimizer
            }
            path = self._snapshot_manager.save(
                step=self._update_counter,
                model=self.model,
                optimizers=optimizers
            )
            log.info(f"Saved snapshot {path}")
            # Prune old snapshots (keep last 5)
            self._snapshot_manager.prune(keep_last=5)
        except Exception as e:
            log.warning(f"Failed saving snapshot: {e}")

