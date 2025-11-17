
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import threading
import time
import csv
from datetime import datetime
# (removed unused imports math, random)

import numpy as np
import torch
import torch.nn.functional as F

from snake_sim.rl.training.rl_data_queue import RLMetaDataQueue
from snake_sim.rl.types import RLTransitionData, PPOMetaData, State
from snake_sim.rl.snapshot_manager import SnapshotManager
from snake_sim.rl.state_builder import print_channel
from snake_sim.rl.models.ppo_model import model_factory

log = logging.getLogger(Path(__file__).stem)

rl_data_queue = RLMetaDataQueue()


@dataclass
class PPOTrainerConfig:
    gamma: float = 0.99
    lam: float = 0.95  # GAE lambda
    clip_range: float = 0.3  # INCREASED from 0.2 - allow bigger policy updates
    policy_lr: float = 1e-4  # LOWERED for stability - prevent policy collapse
    value_lr: float = 3e-4   # LOWERED for stability - was destroying learned values
    epochs: int = 1  # Single epoch to prevent catastrophic forgetting
    minibatch_size: int = 1024  # Reduced for more frequent updates
    entropy_coef: float = 0.02  # INCREASED from 0.01 - need some exploration
    value_coef: float = 0.5     # Reduced from 1.0 to stabilize value function
    max_grad_norm: float = 0.5
    device: str = 'auto'
    snapshot_dir: Optional[str] = None  # directory of model snapshots
    snapshot_base_name: str = 'ppo_model'
    save_every_updates: int = 5  # save snapshot after this many successful updates
    # Adaptive exploration - DISABLED by setting unreachable patience
    stagnation_patience: int = 100000  # Effectively disabled
    exploration_boost_factor: float = 1.5
    max_entropy_coef: float = 0.05  # REDUCED - don't let it get too random
    # Policy collapse prevention
    # KL / policy stability controls
    kl_target: float = 0.8  # Relaxed (was 0.5) to allow larger early improvements
    enable_early_stopping: bool = True  # Stop epochs early if policy changes too much
    kl_warmup_updates: int = 50  # Shorter warmup (was 100) so stability checks engage sooner
    min_return_threshold: Optional[float] = None  # Rollback if returns drop below this
    # Performance / acceleration toggles
    use_amp: bool = True  # Mixed precision training (CUDA only)
    compile_model: bool = True  # torch.compile for model speedup (PyTorch 2+)


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

        # Mixed precision setup (only if CUDA available & enabled)
        self._amp_enabled = (self.device.type == 'cuda')
        # Use new torch.amp API (compatible with deprecation warnings)
        try:
            from torch import amp as _torch_amp  # PyTorch 2 style
            self._scaler = _torch_amp.GradScaler('cuda', enabled=(self._amp_enabled and self.cfg.use_amp))
        except Exception:
            # Fallback for older versions (shouldn't happen, but safe)
            self._scaler = torch.cuda.amp.GradScaler(enabled=(self._amp_enabled and self.cfg.use_amp))
        # Track if we auto-disable AMP due to runtime dtype issues
        self._amp_runtime_disabled = False
        
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
        
        # Stagnation detection for adaptive exploration
        self._best_return = float('-inf')
        self._updates_since_improvement = 0
        self._exploration_boosted = False
        
        # Value normalization for stability
        self._return_mean = 0.0
        self._return_std = 1.0
        self._return_count = 0
        self._enable_value_norm = True  # Normalize returns for value function training
        
        # CSV logging for training statistics
        self._csv_file = None
        self._csv_writer = None
        self._setup_csv_logging()
        
        log.info(f"PPOTrainer initialized device={self.device}")
    
    def _setup_csv_logging(self):
        """Setup CSV file for logging training statistics."""
        if self._snapshot_dir:
            csv_dir = Path(self._snapshot_dir)
        else:
            csv_dir = Path("models/training_logs")
        
        csv_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = csv_dir / f"training_stats_{timestamp}.csv"
        
        try:
            self._csv_file = open(csv_path, 'w', newline='')
            self._csv_writer = csv.writer(self._csv_file)
            
            # Write header
            header = [
                'timestamp',
                'update',
                'policy_loss',
                'value_loss',
                'entropy',
                'kl_div',
                'returns_mean',
                'advantages_mean',
                'batch_size',
                'epochs_completed',
                'entropy_coef',
                'exploration_boosted',
                'best_return',
                'updates_since_improvement'
            ]
            self._csv_writer.writerow(header)
            self._csv_file.flush()
            
            log.info(f"📝 Training statistics will be logged to: {csv_path}")
        except Exception as e:
            log.warning(f"Failed to setup CSV logging: {e}")
            self._csv_file = None
            self._csv_writer = None
    
    def _log_to_csv(self, stats: dict):
        """Log training statistics to CSV file.
        
        Args:
            stats: Dictionary containing training statistics
        """
        if self._csv_writer is None:
            return
        
        try:
            row = [
                datetime.now().isoformat(),
                stats.get('update', ''),
                stats.get('policy_loss', ''),
                stats.get('value_loss', ''),
                stats.get('entropy', ''),
                stats.get('kl_div', ''),
                stats.get('returns_mean', ''),
                stats.get('advantages_mean', ''),
                stats.get('batch_size', ''),
                stats.get('epochs_completed', ''),
                stats.get('entropy_coef', ''),
                stats.get('exploration_boosted', ''),
                self._best_return,
                self._updates_since_improvement
            ]
            self._csv_writer.writerow(row)
            
            # Flush every 10 updates to ensure data is written
            if stats.get('update', 0) % 10 == 0:
                self._csv_file.flush()
        except Exception as e:
            log.warning(f"Failed to log to CSV: {e}")

    def _ensure_model(self, first_state: State):
        """Create model lazily from first state to determine input dimensions."""
        if self.model is not None:
            return
        
        ctx_dim = 0 if first_state.ctx is None else first_state.ctx.shape[0]
        in_channels = first_state.map.shape[0]
        self.model = model_factory(in_channels, ctx_dim, map_size=first_state.map.shape[1]).to(self.device)
        # Optional compile for speed (falls back gracefully)
        if self.cfg.compile_model and self.device.type == 'cuda':
            try:
                self.model = torch.compile(self.model)
                log.info("Model compiled with torch.compile()")
            except Exception as e:
                log.warning(f"torch.compile failed; continuing without compile: {e}")

        # Split optimizers using param grouping if available
        if hasattr(self.model, 'actor_parameters') and hasattr(self.model, 'critic_parameters'):
            self.policy_optimizer = torch.optim.Adam(self.model.actor_parameters(), lr=self.cfg.policy_lr, eps=1e-5)
            self.value_optimizer = torch.optim.Adam(self.model.critic_parameters(), lr=self.cfg.value_lr, eps=1e-5)
            log.info("Initialized split optimizers (actor & critic)")
        else:
            # Fallback to unified parameters (should not occur after model patch)
            self.policy_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.policy_lr, eps=1e-5)
            self.value_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.value_lr, eps=1e-5)
            log.warning("Model missing parameter grouping helpers; using unified optimizers")
        
        # Setup snapshot manager if directory provided
        if self._snapshot_dir:
            def model_creator():
                return model_factory(in_channels, ctx_dim, map_size=first_state.map.shape[1])
            self._snapshot_manager = SnapshotManager(
                dir=self._snapshot_dir,
                base_name=self.cfg.snapshot_base_name,
                factory=model_creator
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
            log.info(f"  Exploration: entropy_coef={self.cfg.entropy_coef:.4f} | Best Return: {self._best_return:.3f}")
            log.info(f"  Stagnation: {self._updates_since_improvement} updates without improvement")
            
            # Learning indicators
            if returns_trend > 0.01:
                log.info("  🚀 Snakes showing improvement in returns!")
            elif returns_trend < -0.01:
                log.info("  📉 Returns declining - possible overfitting or poor exploration")
            
            if entropy_avg < 0.5:
                log.info("  ⚠️  Low entropy - snakes may be too deterministic")
            elif entropy_avg > 1.5:
                log.info("  🎲 High entropy - snakes exploring actively")
            
            # Stagnation warning
            patience_remaining = self.cfg.stagnation_patience - self._updates_since_improvement
            if patience_remaining <= 10 and patience_remaining > 0 and not self._exploration_boosted:
                log.warning(f"  ⏰ Stagnation warning: {patience_remaining} updates until exploration boost")

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
    
    def reset_stagnation_tracking(self):
        """Manually reset stagnation tracking. Useful after major changes."""
        self._best_return = float('-inf')
        self._updates_since_improvement = 0
        log.info("🔄 Stagnation tracking reset")
    
    def boost_exploration(self, factor: float = None):
        """Manually boost exploration by increasing entropy coefficient.
        
        Args:
            factor: Multiplication factor for entropy_coef. If None, uses config default.
        """
        if factor is None:
            factor = self.cfg.exploration_boost_factor
        
        old_entropy = self.cfg.entropy_coef
        self.cfg.entropy_coef *= factor
        self._exploration_boosted = True
        log.info(f"🎲 Manual exploration boost: {old_entropy:.4f} → {self.cfg.entropy_coef:.4f}")
    
    def reset_exploration(self, base_entropy: float = 0.1):
        """Reset exploration to a base level.
        
        Args:
            base_entropy: The base entropy coefficient to reset to.
        """
        old_entropy = self.cfg.entropy_coef
        self.cfg.entropy_coef = base_entropy
        self._exploration_boosted = False
        log.info(f"↩️  Exploration reset: {old_entropy:.4f} → {self.cfg.entropy_coef:.4f}")

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
        # Optional action masks (state.meta['action_mask'] expected as np.array shape (4,), 1 valid / 0 invalid)
        masks = None
        if all((s.meta is not None and 'action_mask' in s.meta) for s in states) and len(states) > 0:
            try:
                masks_np = [s.meta['action_mask'] for s in states]
                masks = torch.from_numpy(np.stack(masks_np)).float().to(self.device)
            except Exception as e:
                log.warning(f"Failed to stack action masks; ignoring masks. Error: {e}")
                masks = None
        batch = {
            'states': state_batch,
            'actions': actions,
            'old_log_probs': old_log_probs,
            'advantages': advantages,
            'returns': returns
        }
        if masks is not None:
            batch['action_mask'] = masks
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
        """PPO update with multiple epochs through the data."""
        # Set model to training mode for gradient computation
        self.model.train()
        
        states = batch['states']
        actions = batch['actions']
        old_log_probs = batch['old_log_probs']
        advantages = batch['advantages']
        returns = batch['returns']
        action_mask_full = batch.get('action_mask', None)
        
        batch_size = len(actions)
        
        # Update return normalization statistics
        if self._enable_value_norm:
            returns_np = returns.cpu().numpy()
            self._update_return_stats(returns_np)
            # Normalize returns for value function training
            normalized_returns = (returns - self._return_mean) / (self._return_std + 1e-8)
        else:
            normalized_returns = returns
        
        # Track cumulative metrics across all epochs
        epoch_policy_losses = []
        epoch_value_losses = []
        epoch_entropies = []
        epoch_kl_divs = []
        
        # Train for multiple epochs on the same batch
        for epoch in range(self.cfg.epochs):
            # Shuffle indices each epoch for better training
            indices = torch.randperm(batch_size, device=self.device)
            
            # Process the entire batch (could split into minibatches here if needed)
            # For now, we'll process in one go but with shuffled order
            shuffled_states = self._shuffle_state_batch(states, indices)
            shuffled_actions = actions[indices]
            shuffled_old_log_probs = old_log_probs[indices]
            shuffled_advantages = advantages[indices]
            shuffled_returns = normalized_returns[indices]  # Use normalized returns
            shuffled_action_mask = action_mask_full[indices] if action_mask_full is not None else None
            
            # Forward + loss under optional mixed precision
            # Autocast with new API if available
            try:
                from torch import amp as _torch_amp
                autocast_ctx = _torch_amp.autocast('cuda', enabled=(self._amp_enabled and self.cfg.use_amp))
            except Exception:
                autocast_ctx = torch.cuda.amp.autocast(enabled=(self._amp_enabled and self.cfg.use_amp))
            with autocast_ctx:
                logits, values = self._forward(shuffled_states)
            # Convert to float32 for stable loss math regardless of autocast dtype
            logits = logits.float()
            values = values.float()
            if shuffled_action_mask is not None:
                # Use moderate negative to avoid fp16 overflow and maintain gradient stability post-cast
                logits = logits.masked_fill(shuffled_action_mask == 0, -60.0)
            dist = torch.distributions.Categorical(logits=logits)
            log_probs = dist.log_prob(shuffled_actions).float()
            entropy = dist.entropy().mean().float()

            # Compute KL divergence to detect policy collapse
            with torch.no_grad():
                kl_div = (shuffled_old_log_probs - log_probs).mean()
                approx_kl = kl_div.item()
            
            # Early stopping if policy is changing too much (skip during warmup)
            in_warmup = self._update_counter < self.cfg.kl_warmup_updates
            if self.cfg.enable_early_stopping and not in_warmup and approx_kl > self.cfg.kl_target * 1.5:
                log.warning(f"⚠️  Early stopping at epoch {epoch}/{self.cfg.epochs}: KL={approx_kl:.4f} > target {self.cfg.kl_target*1.5:.4f}")
                log.warning(f"   Policy changing too fast - stopping to prevent collapse")
                break
            elif in_warmup and approx_kl > 2.0:
                log.info(f"🔥 Warmup phase ({self._update_counter}/{self.cfg.kl_warmup_updates}): Large KL={approx_kl:.4f} allowed")

            ratio = torch.exp(log_probs - shuffled_old_log_probs)
            surr1 = ratio * shuffled_advantages
            surr2 = torch.clamp(ratio, 1 - self.cfg.clip_range, 1 + self.cfg.clip_range) * shuffled_advantages
            policy_loss = -torch.mean(torch.min(surr1, surr2))
            value_loss = F.mse_loss(values, shuffled_returns)
            
            # Detect value function explosion
            if value_loss.item() > 100.0:
                log.error(f"🚨 VALUE FUNCTION EXPLOSION: loss={value_loss.item():.2f}")
                log.error(f"   Value predictions - min: {values.min().item():.2f}, max: {values.max().item():.2f}, mean: {values.mean().item():.2f}")
                log.error(f"   Return targets - min: {shuffled_returns.min().item():.2f}, max: {shuffled_returns.max().item():.2f}, mean: {shuffled_returns.mean().item():.2f}")
                log.error(f"   Return normalization - mean: {self._return_mean:.2f}, std: {self._return_std:.2f}")
                log.error(f"   Skipping this update to prevent collapse")
                return None
            
            # Simple stable loss combination
            total_loss = policy_loss + self.cfg.value_coef * value_loss - self.cfg.entropy_coef * entropy
            
            try:
                self.policy_optimizer.zero_grad(set_to_none=True)
                self.value_optimizer.zero_grad(set_to_none=True)
                if self._amp_enabled and self.cfg.use_amp and not self._amp_runtime_disabled:
                    self._scaler.scale(total_loss).backward()
                    self._scaler.unscale_(self.policy_optimizer)
                    self._scaler.unscale_(self.value_optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                    self._scaler.step(self.policy_optimizer)
                    self._scaler.step(self.value_optimizer)
                    self._scaler.update()
                else:
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                    self.policy_optimizer.step()
                    self.value_optimizer.step()
                # Track metrics
                epoch_policy_losses.append(policy_loss.item())
                epoch_value_losses.append(value_loss.item())
                epoch_entropies.append(entropy.item())
                epoch_kl_divs.append(approx_kl)
            except RuntimeError as e:
                if 'expected Half' in str(e) and not self._amp_runtime_disabled:
                    log.warning(f"AMP dtype mismatch detected; disabling AMP for future updates: {e}")
                    self._amp_runtime_disabled = True
                    # Retry in full precision
                    self.policy_optimizer.zero_grad(set_to_none=True)
                    self.value_optimizer.zero_grad(set_to_none=True)
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                    self.policy_optimizer.step()
                    self.value_optimizer.step()
                    epoch_policy_losses.append(policy_loss.item())
                    epoch_value_losses.append(value_loss.item())
                    epoch_entropies.append(entropy.item())
                    epoch_kl_divs.append(approx_kl)
                elif "inplace operation" in str(e):
                    log.warning(f"Gradient computation failed due to inplace operation, skipping epoch {epoch}: {e}")
                    continue
                else:
                    raise

        # If all epochs failed, return None
        if not epoch_policy_losses:
            return None
            
        self._update_counter += 1
        
        # Adaptive exploration: check for stagnation
        current_return = float(returns.mean().item())
        self._check_stagnation_and_adapt(current_return)
        
        # Save snapshot periodically
        if self._snapshot_manager and (self._update_counter % self.cfg.save_every_updates == 0):
            self._save_snapshot()

        # Return average metrics across all epochs
        stats = {
            'policy_loss': float(np.mean(epoch_policy_losses)),
            'value_loss': float(np.mean(epoch_value_losses)),
            'entropy': float(np.mean(epoch_entropies)),
            'kl_div': float(np.mean(epoch_kl_divs)),
            'update': self._update_counter,
            'advantages_mean': float(advantages.mean().item()),
            'returns_mean': float(returns.mean().item()),
            'batch_size': batch_size,
            'epochs_completed': len(epoch_policy_losses),
            'entropy_coef': self.cfg.entropy_coef,  # Track current entropy coefficient
            'exploration_boosted': self._exploration_boosted
        }
        
        # Log to CSV file
        self._log_to_csv(stats)
        
        # Log detailed metrics with trends
        self._log_detailed_metrics(stats)
        
        return stats
    
    def _update_return_stats(self, returns: np.ndarray):
        """Update running statistics for return normalization.
        
        Uses Welford's online algorithm for numerical stability.
        
        Args:
            returns: Array of return values from current batch
        """
        batch_mean = returns.mean()
        batch_std = returns.std()
        batch_count = len(returns)
        
        # Welford's online algorithm for running mean/std
        if self._return_count == 0:
            self._return_mean = batch_mean
            self._return_std = max(batch_std, 1.0)  # Prevent division by zero
        else:
            # Combine old and new statistics
            total_count = self._return_count + batch_count
            delta = batch_mean - self._return_mean
            self._return_mean += delta * batch_count / total_count
            
            # Update std (simplified version)
            self._return_std = 0.99 * self._return_std + 0.01 * batch_std
            self._return_std = max(self._return_std, 1.0)  # Floor at 1.0
        
        self._return_count += batch_count
        
        # Log normalization stats occasionally
        if self._return_count % 10000 < batch_count:
            log.info(f"📊 Return normalization: mean={self._return_mean:.3f}, std={self._return_std:.3f}")
    
    def _shuffle_state_batch(self, state_batch, indices):
        """Shuffle a state batch according to given indices.
        
        Args:
            state_batch: Either a tensor (B,C,H,W) or dict with 'map' and 'ctx' keys
            indices: Tensor of shuffled indices
        
        Returns:
            Shuffled state batch in the same format as input
        """
        if isinstance(state_batch, dict):
            return {
                'map': state_batch['map'][indices],
                'ctx': state_batch['ctx'][indices]
            }
        else:
            return state_batch[indices]
    
    def _check_stagnation_and_adapt(self, current_return: float):
        """Check if training is stagnating and adapt exploration if needed.
        
        Args:
            current_return: Current average return from the batch
        """
        # Track if we've improved
        improvement_threshold = 0.01  # Consider it improvement if return increases by at least 1%
        
        if current_return > self._best_return * (1.0 + improvement_threshold):
            # Significant improvement detected
            self._best_return = current_return
            self._updates_since_improvement = 0
            
            # If exploration was boosted and we're improving, we can gradually reduce it
            if self._exploration_boosted:
                # Gradually return to normal exploration
                base_entropy = 0.1  # Original baseline
                self.cfg.entropy_coef = max(base_entropy, self.cfg.entropy_coef * 0.95)
                
                # Check if we've returned to baseline
                if self.cfg.entropy_coef <= base_entropy * 1.05:
                    self._exploration_boosted = False
                    self.cfg.entropy_coef = base_entropy
                    log.info(f"✅ Exploration returned to baseline: entropy_coef={self.cfg.entropy_coef:.4f}")
        else:
            # No improvement
            self._updates_since_improvement += 1
            
            # Check if we should boost exploration
            if (not self._exploration_boosted and 
                self._updates_since_improvement >= self.cfg.stagnation_patience):
                
                # Boost exploration (cap at max_entropy_coef)
                old_entropy = self.cfg.entropy_coef
                self.cfg.entropy_coef = min(
                    self.cfg.entropy_coef * self.cfg.exploration_boost_factor,
                    self.cfg.max_entropy_coef
                )
                self._exploration_boosted = True
                self._updates_since_improvement = 0  # Reset counter
                
                log.warning(f"🔄 STAGNATION DETECTED after {self.cfg.stagnation_patience} updates!")
                log.warning(f"   Best return: {self._best_return:.3f} | Current: {current_return:.3f}")
                log.warning(f"   Boosting exploration: entropy_coef {old_entropy:.4f} → {self.cfg.entropy_coef:.4f}")
                log.warning(f"   Snakes will explore more aggressively now!")
            
            # If already boosted and still stagnating, keep boosting (with cap)
            elif (self._exploration_boosted and 
                  self._updates_since_improvement >= self.cfg.stagnation_patience):
                
                # Progressive boost (more frequent, smaller increments)
                old_entropy = self.cfg.entropy_coef
                self.cfg.entropy_coef = min(
                    self.cfg.entropy_coef * 1.2,  # Smaller increments
                    self.cfg.max_entropy_coef
                )
                self._updates_since_improvement = 0  # Reset counter to keep trying
                
                if self.cfg.entropy_coef >= self.cfg.max_entropy_coef:
                    log.error(f"🚨 MAXIMUM EXPLORATION REACHED!")
                    log.error(f"   Entropy: {old_entropy:.4f} → {self.cfg.entropy_coef:.4f}")
                    log.error(f"   🔴 REWARD FUNCTION MAY BE THE PROBLEM!")
                    log.error(f"   Best return achieved: {self._best_return:.3f}, current: {current_return:.3f}")
                    log.error(f"   Consider: 1) Reward shaping, 2) Curriculum learning, 3) Reset from best checkpoint")
                else:
                    log.warning(f"⚠️  CONTINUED STAGNATION - Incrementing exploration")
                    log.warning(f"   Entropy: {old_entropy:.4f} → {self.cfg.entropy_coef:.4f}")

    # ---- Background Thread API ----
    def start_background(self, interval_sec: float = 0.5):
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
                        
                        # Detect policy collapse
                        if stats['returns_mean'] < -10.0:
                            log.error(f"🚨 POLICY COLLAPSE WARNING: Return={stats['returns_mean']:.3f}")
                        
                        # Brief update log
                        log.info(f"Update {stats['update']}: "
                               f"P_Loss={stats['policy_loss']:.3f}, "
                               f"V_Loss={stats['value_loss']:.3f}, "
                               f"Return={stats['returns_mean']:.3f}, "
                               f"KL={stats.get('kl_div', 0):.4f}, "
                               f"Batch={stats['batch_size']}")
                        
                        # Detect concerning patterns
                        if stats.get('kl_div', 0) > self.cfg.kl_target:
                            log.warning(f"⚠️  High KL divergence: {stats['kl_div']:.4f} - policy changing rapidly")
                        
                        if stats['value_loss'] > 10.0:
                            log.warning(f"⚠️  High value loss: {stats['value_loss']:.3f} - critic may be unstable")
                        
                        # Detailed stats less frequently 
                        if current_time - last_stats_time >= stats_interval:
                            queue_size = rl_data_queue.size()
                            log.info(f"📊 Training Overview (Update {stats['update']}):")
                            log.info(f"   Queue Size: {queue_size} | Batch Size: {stats['batch_size']}")
                            log.info(f"   Policy Loss: {stats['policy_loss']:.4f} | Value Loss: {stats['value_loss']:.4f}")
                            log.info(f"   Entropy: {stats['entropy']:.4f} | KL Div: {stats.get('kl_div', 0):.4f}")
                            log.info(f"   Avg Return: {stats['returns_mean']:.4f} | Best: {self._best_return:.4f}")
                            log.info(f"   Epochs: {stats['epochs_completed']}/{self.cfg.epochs}")
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
    
    def close(self):
        """Cleanup resources including CSV file."""
        if self._csv_file:
            try:
                self._csv_file.flush()
                self._csv_file.close()
                log.info("📝 Training statistics CSV file closed")
            except Exception as e:
                log.warning(f"Failed to close CSV file: {e}")
        
        # Stop background thread if running
        if self._bg_running:
            self.stop_background(join=True, timeout=5.0)
    
    def __del__(self):
        """Ensure CSV file is closed on deletion."""
        self.close()
