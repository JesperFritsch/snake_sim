
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
from snake_sim.rl.snapshot_manager import SnapshotManager, SNAPSHOT_BASE_DIR
from snake_sim.rl.state_builder import print_channel
from snake_sim.rl.models.ppo_model import model_factory
from snake_sim.rl.action_masking import apply_action_mask_to_logits

log = logging.getLogger(Path(__file__).stem)

@dataclass
class PPOTrainerConfig:
    gamma: float = 0.99
    lam: float = 0.98  # GAE lambda
    clip_range: float = 0.4  # Slightly wider clip -> larger effective updates
    # Learning rates: the previous defaults produced near-zero policy updates
    # (ratio_std ~ 1e-4, clipfrac ~ 0). These values are still modest for PPO,
    # but should yield meaningfully non-zero updates.
    policy_lr: float = 5e-4
    # Critic can become unstable fast in sparse/noisy-reward settings.
    # Keep value_lr lower than policy_lr unless you have very reliable return targets.
    value_lr: float = 1e-3
    minibatch_size: int = 256  # More minibatches/rollout -> more gradient steps
    rollout_size: int = 8192  # INCREASED - collect more before training (was 2048)
    train_epochs: int = 2  # More optimizer steps per rollout -> faster learning
    entropy_coef: float = 0.015  # Slightly lower entropy to reduce overly random updates
    value_coef: float = 0.5     # Reduced from 1.0 to stabilize value function
    max_grad_norm: float = 1.0  # Increased from 0.5 (double norm fix allows higher safety margin)
    device: str = 'auto'
    snapshot_base_name: str = 'ppo_model'
    save_every_updates: int = 1  # save snapshot after this many successful updates
    # Keep an unpruned archive snapshot every N updates (in <snapshot_dir>/archive/)
    archive_every_n_updates: int = 2000
    # Adaptive exploration - DISABLED by setting unreachable patience
    stagnation_patience: int = 100000  # Effectively disabled
    exploration_boost_factor: float = 1.5
    max_entropy_coef: float = 0.05  # REDUCED - don't let it get too random
    # Policy collapse prevention
    # KL / policy stability controls
    kl_target: float = 0.1  # Lower target; used for warnings (approximate KL proxy)
    enable_early_stopping: bool = False  # Disabled - was causing oscillation
    # Hard cap to avoid catastrophic policy updates. If approx_kl exceeds this,
    # skip the update and do NOT publish a snapshot.
    kl_hard_limit: float = 0.5
    # Only train on complete episodes (recommended for episodic tasks)
    require_complete_episodes: bool = False

    # If > 0: only train on trajectories with at least this many transitions.
    # Useful when using heterogeneous env groups where some snakes contribute
    # very short fragments in the queue (high variance advantages).
    min_trajectory_len: int = 512 # same as mini batch size

    # Performance / acceleration toggles
    use_amp: bool = True  # Mixed precision training (CUDA only)
    compile_model: bool = False  # torch.compile (not supported on Python 3.14+)
    vf_clip_range: float = 0.2

    # Auxiliary loss: encourage the *raw* (pre-mask) policy logits to assign low
    # preference to invalid actions, so the model internalizes action legality.
    # Keep this small; masking still enforces safety at sampling-time.
    illegal_logit_coef: float = 0.02
    # We use softplus(logit - margin). With margin=0, it pushes invalid logits < 0.
    illegal_logit_margin: float = 0.0


class PPOTrainer:
    """Minimal PPO trainer that creates its own model lazily from first batch of data."""

    def __init__(self, transition_queue: RLMetaDataQueue, config: PPOTrainerConfig = None, snapshot_dir: str = None):
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
        self.optimizer = None
        # Some test/scratch usage constructs PPOTrainer without an explicit snapshot_dir.
        # Fall back to the configured base name so CSV/snapshots have a stable location.
        self._snapshot_dir = snapshot_dir or self.cfg.snapshot_base_name
        self._snapshot_manager = None
        self._adv_returns = {}
        self._train_lock = threading.Lock()
        self._bg_thread = None
        self._bg_running = False
        self._transition_queue = transition_queue 

        # Explicit behavior metric (comes from transition.ate_food)
        self._last_batch_foods_eaten = 0
        
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
        # Normalizing critic targets is one of the safest stabilizers.
        # This implementation uses Welford updates (no exponential decay), so it should be stable.
        self._enable_value_norm = True
        
        # CSV logging for training statistics
        self._csv_file = None
        self._csv_writer = None
        self._setup_csv_logging()

        self.accumulating_trajs: Dict[Tuple[str, str, int], List[RLTransitionData]] = {}

        log.info(f"PPOTrainer initialized device={self.device}")

    def _setup_csv_logging(self):
        """Setup CSV file for logging training statistics."""
        csv_dir = Path(SNAPSHOT_BASE_DIR, self._snapshot_dir)
        
        csv_dir.mkdir(parents=True, exist_ok=True)

        # Reuse an existing stats file if present, so restarts don't create many CSVs
        # for what is effectively the same training run (same snapshot_dir).
        #
        # Strategy:
        #   - If any training_stats_*.csv exists, pick the most recently modified and append.
        #   - Otherwise create a new timestamped file.
        #
        # This keeps one evolving log file per snapshot_dir by default.
        existing = sorted(csv_dir.glob("training_stats.csv"), key=lambda p: p.stat().st_mtime)
        if existing:
            csv_path = existing[-1]
            mode = 'a'
        else:
            csv_path = csv_dir / f"training_stats.csv"
            mode = 'w'
        
        try:
            self._csv_file = open(csv_path, mode, newline='')
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
                'updates_since_improvement',
                'value_pred_denorm_mean',
                'value_residual_mean',
                # PPO diagnostics
                'clipfrac',
                'ratio_mean',
                'ratio_std',
                'approx_kl',
                'explained_variance',
                'invalid_action_frac',
                'illegal_logit_loss',
                # Model diagnostics
                'path_mixer_raw',
                'path_mixer_alpha',
                # Explicit behavior metrics
                'foods_eaten',
                'foods_per_1k_steps',
                'steps_per_food'
            ]

            # Only write header if file is new/empty (or doesn't already have a header).
            try:
                needs_header = False
                if mode == 'w':
                    needs_header = True
                else:
                    # Append mode: check if file seems empty or lacks the expected header.
                    if csv_path.stat().st_size == 0:
                        needs_header = True
                    else:
                        with open(csv_path, 'r', newline='') as rf:
                            first_line = rf.readline().strip()
                        expected_first = ','.join(header)
                        if first_line != expected_first:
                            # If the header differs (older format), don't write a second header.
                            # We keep appending rows to that file for continuity.
                            needs_header = False
                if needs_header:
                    self._csv_writer.writerow(header)
                    self._csv_file.flush()
            except Exception as e:
                log.warning(f"Failed header check/write for CSV {csv_path}: {e}")

            log.info(f"📝 Training statistics will be logged to: {csv_path} (mode={mode})")
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
                self._updates_since_improvement,
                stats.get('value_pred_denorm_mean', ''),
                stats.get('value_residual_mean', ''),
                # PPO diagnostics
                stats.get('clipfrac', ''),
                stats.get('ratio_mean', ''),
                stats.get('ratio_std', ''),
                stats.get('approx_kl', ''),
                stats.get('explained_variance', ''),
                stats.get('invalid_action_frac', ''),
                stats.get('illegal_logit_loss', ''),
                # Model diagnostics
                stats.get('path_mixer_raw', ''),
                stats.get('path_mixer_alpha', ''),
                # Explicit behavior metrics
                stats.get('foods_eaten', ''),
                stats.get('foods_per_1k_steps', ''),
                stats.get('steps_per_food', ''),
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
        # Assume ctx always present per new state design.
        if first_state.ctx is None:
            raise ValueError("State ctx missing; new design requires ctx tensor")
        ctx_dim = first_state.ctx.shape[0]
        in_channels = first_state.map.shape[0]
        self.model = model_factory(in_channels, ctx_dim).to(self.device)
        # Optional compile for speed (falls back gracefully)
        if self.cfg.compile_model and self.device.type == 'cuda':
            try:
                self.model = torch.compile(self.model)
                log.info("Model compiled with torch.compile()")
            except Exception as e:
                log.warning(f"torch.compile failed; continuing without compile: {e}")

        # Some PyTorch versions return a wrapper module from torch.compile that owns the
        # compiled graph but stores parameters on an underlying original module.
        # For optimizers and state_dict operations we want to target the underlying module
        # when available.
        core_model = getattr(self.model, '_orig_mod', self.model)
        # Separate optimizer parameter groups: policy vs value head for distinct learning rates.
        value_params = [p for n, p in core_model.named_parameters() if 'value_head' in n]
        policy_params = [p for n, p in core_model.named_parameters() if 'value_head' not in n]
        self.optimizer = torch.optim.Adam([
            {'params': policy_params, 'lr': self.cfg.policy_lr},
            {'params': value_params, 'lr': self.cfg.value_lr}
        ], eps=1e-5)
        log.info(f"Initialized optimizer with policy_lr={self.cfg.policy_lr} value_lr={self.cfg.value_lr}")
        
        def model_creator():
            return model_factory(in_channels, ctx_dim)
        self._snapshot_manager = SnapshotManager(
            dir_name=self._snapshot_dir,
            base_name=self.cfg.snapshot_base_name,
            factory=model_creator,
            archive_every_n=self.cfg.archive_every_n_updates,
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
            values.clear()  # Clear after computing averages to save memory
        return averages

    def _get_policy_mixer_stats(self) -> dict:
        """Return pathway mixer diagnostics.

        Intentionally assumes required fields exist (crash loudly if they don't).
        """
        with torch.no_grad():
            mixer_raw = float(self.model.pathway_mixer.detach().cpu().item())
            alpha = float(torch.sigmoid(self.model.pathway_mixer.detach()).cpu().item())
        return {
            'path_mixer_raw': mixer_raw,
            'path_mixer_alpha': alpha,
        }

    def _log_detailed_metrics(self, stats: dict):
        """Log detailed training metrics with trends and running averages."""
        # Add model-specific diagnostics (if available)
        stats.update(self._get_policy_mixer_stats())

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

            log.info(
                f"  Mixer: raw={stats['path_mixer_raw']:+.3f} | "
                f"alpha(sigmoid)={stats['path_mixer_alpha']:.3f} "
                f"(0=safety, 1=spatial)"
            )

            # PPO diagnostics (helps answer: "are we actually updating?")
            if 'clipfrac' in stats:
                log.info(
                    "  PPO: clipfrac=%.3f | ratio(mean±std)=%.3f±%.3f | approx_kl=%.6f | EV=%.3f | invalid=%.3f",
                    stats.get('clipfrac', float('nan')),
                    stats.get('ratio_mean', float('nan')),
                    stats.get('ratio_std', float('nan')),
                    stats.get('approx_kl', float('nan')),
                    stats.get('explained_variance', float('nan')),
                    stats.get('invalid_action_frac', float('nan')),
                )
            
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
            # If we're compiled, make sure weights load into the original module.
            target_model = getattr(self.model, '_orig_mod', self.model)
            success = self._snapshot_manager.reload_into(target_model)
            if success and self._snapshot_manager._last_loaded:
                loaded_info = self._snapshot_manager._last_loaded
                self._update_counter = loaded_info.step
                log.debug(f"📁 Loaded snapshot from {loaded_info.path}, resuming from step {self._update_counter}")
                # Also load optimizer states if available
                data = torch.load(loaded_info.path, map_location=self.device)
                if isinstance(data, dict):
                    if 'optimizer' in data and self.optimizer:
                        self.optimizer.load_state_dict(data['optimizer'])
                        log.debug("📈 Restored optimizer state")
        except Exception as e:
            log.warning(f"Failed to load snapshot: {e}")

    def get_queue_size(self):
        """Get current transition queue size."""
        return self._transition_queue.size()

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

        Proper PPO workflow:
        1. Collect complete episodes until we have rollout_size transitions
        2. Compute advantages for each complete episode
        3. Train on the collected data for multiple epochs
        4. Split data into minibatches during training for efficiency

        Returns training stats dict or None if insufficient data.
        """
        with self._train_lock:
            if self._transition_queue.size() < self.cfg.rollout_size:
                return None  # Not enough yet – keep accumulating
            # _adv_returns is only needed for collating the *current* rollout batch.
            # Keeping old entries causes unbounded growth (memory leak) in long runs.
            self._adv_returns.clear()
            transitions = self._transition_queue.get_transitions()
            if not transitions:
                return None
            
            # Initialize model lazily from first transition
            self._ensure_model(transitions[0].state)
            
            # Group transitions by (snake_id, episode_id, process_id) to form trajectories
            for t in transitions:
                snake_id = t.snake_id
                episode_id = t.episode_id
                process_id = t.process_id
                self.accumulating_trajs.setdefault((snake_id, episode_id, process_id), []).append(t)
            
            
            all_ready_trajs: Dict[Tuple[str, str, int], List[RLTransitionData]] = {}
            keys_to_remove: List[Tuple[str, str, int]] = []

            # We collect keys to remove and delete them after the loop.
            for key, traj in self.accumulating_trajs.items():
                if not traj:
                    keys_to_remove.append(key)
                    continue

                # Gate readiness based on configured minimum length / completion.
                if self.cfg.min_trajectory_len > 0 and len(traj) < self.cfg.min_trajectory_len and not traj[-1].done:
                    continue  # let short trajs accumulate more
                if self.cfg.require_complete_episodes and not traj[-1].done:
                    continue  # wait for episode completion

                # Sort in-place for stable GAE + contiguous check.
                traj.sort(key=lambda x: x.transition_nr)

                if not all(b.transition_nr == a.transition_nr + 1 for a, b in zip(traj, traj[1:])):
                    log.warning(
                        "Trajectory has non-sequential transitions: snake_id=%s, episode_id=%s, process_id=%s",
                        key[0],
                        key[1],
                        key[2],
                    )
                    # Keep accumulating; we don't want to train on a broken sequence.
                    # (If you *do* want to train anyway, change this to still mark ready.)
                    continue

                all_ready_trajs[key] = traj
                keys_to_remove.append(key)

            for key in keys_to_remove:
                self.accumulating_trajs.pop(key, None)

            for traj in all_ready_trajs.values():
                snake_id = traj[0].snake_id
                episode_id = traj[0].episode_id
                process_id = traj[0].process_id
                start_step = traj[0].transition_nr
                is_done = traj[-1].done
                traj_length = len(traj)
                log.info(
                    "Processing trajectory: snake_id=%s, episode_id=%s, process=%s, length=%s, start_step=%s, done=%s",
                    snake_id,
                    episode_id,
                    process_id,
                    traj_length,
                    start_step,
                    is_done
                )

            # Compute GAE advantages for each trajectory
            for traj in all_ready_trajs.values():
                self._compute_gae(traj)
            
            # Flatten all trajectories into one list for training
            processed = [t for traj in all_ready_trajs.values() for t in traj]

            # Allow training on smaller batches if we have at least one full trajectory.
            # This prevents throwing away useful data when it's just under the configured minibatch size.
            if len(processed) == 0:
                log.debug("No transitions available to train on yet")
                return None
            
            log.info(f"Training on {len(processed)} transitions from {len(all_ready_trajs)} complete episodes")

            # Count explicit food events in this training batch.
            # This is robust to reward shaping changes.
            try:
                self._last_batch_foods_eaten = int(sum(1 for t in processed if getattr(t, 'ate_food', False)))
            except Exception:
                self._last_batch_foods_eaten = 0
            
            # Train for multiple epochs on the collected data
            stats = None
            for epoch in range(self.cfg.train_epochs):
                # Shuffle data each epoch
                np.random.shuffle(processed)
                
                # Split into minibatches and train
                # Always perform at least one minibatch; if data is scarce, use the whole set once.
                num_minibatches = max(1, len(processed) // self.cfg.minibatch_size)
                for mb_idx in range(num_minibatches):
                    if len(processed) < self.cfg.minibatch_size:
                        # Single smaller batch (use all data)
                        minibatch = processed
                    else:
                        start_idx = mb_idx * self.cfg.minibatch_size
                        end_idx = start_idx + self.cfg.minibatch_size
                        minibatch = processed[start_idx:end_idx]
                    
                    batch = self._collate_batch(minibatch)
                    epoch_stats = self._ppo_update(batch, epoch=epoch)
                    
                    # Keep stats from last minibatch for logging
                    if epoch_stats:
                        stats = epoch_stats
                        stats['epoch'] = epoch
                        stats['minibatch'] = mb_idx
            
            return stats

    # ---- Core Steps ----
    def _compute_gae(self, trajectory: List[RLTransitionData]):
        # Gather trajectory data
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

        # Bootstrap: for truncated (non-terminal) trajectories, bootstrap from V(s_T)
        # using the final transition's next_state.
        #
        # Why: `meta.value_estimate` is V(s_t) for the *current* state at each step.
        # GAE needs V(s_{t+1}) when computing delta_t, and in particular needs V(s_T)
        # for the last step if it is *not* terminal. Reusing V(s_{T-1}) biases the
        # advantages/returns for truncated trajectories.
        bootstrap_value = 0.0
        if values:
            last_t = trajectory[-1]
            if (not last_t.done) and (last_t.next_state is not None):
                # Lazily ensure model exists in odd call paths.
                self._ensure_model(last_t.state)
                try:
                    ns = last_t.next_state
                    if ns.ctx is None:
                        raise ValueError("next_state.ctx missing; required for PPO model")
                    if 'action_features' not in (ns.meta or {}):
                        raise KeyError("next_state.meta missing action_features")

                    map_tensor = torch.from_numpy(np.ascontiguousarray(ns.map)).unsqueeze(0).float().to(self.device, non_blocking=True)
                    ctx_tensor = torch.from_numpy(np.ascontiguousarray(ns.ctx)).unsqueeze(0).float().to(self.device, non_blocking=True)
                    af_tensor = torch.from_numpy(np.ascontiguousarray(ns.meta['action_features'])).unsqueeze(0).float().to(self.device, non_blocking=True)
                    with torch.no_grad():
                        _, v_next = self.model({'map': map_tensor, 'ctx': ctx_tensor, 'action_features': af_tensor})
                    bootstrap_value = float(v_next.squeeze(0).squeeze(-1).detach().cpu().item())
                except Exception as e:
                    # Fall back to a conservative bootstrap of 0.0 (treat as terminal)
                    # rather than silently using V(s_{T-1}).
                    log.warning(f"GAE bootstrap from next_state failed; using 0.0 (treat as terminal): {e}")
                    bootstrap_value = 0.0
            else:
                # Terminal trajectory: V(s_T)=0.0, masking also enforces this.
                bootstrap_value = 0.0
        
        values = np.array(values + [bootstrap_value], dtype=np.float32)
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
        # IMPORTANT: store old value predictions from rollout time for value clipping.
        # PPO value clipping requires comparing to the baseline value predictions that were used
        # when the data was collected (not the current forward pass).
        old_values = torch.tensor([t.meta.value_estimate for t in transitions], dtype=torch.float32, device=self.device)  # type: ignore
        advantages = torch.tensor([self._adv_returns[t.transition_id][0] for t in transitions], dtype=torch.float32, device=self.device)
        returns = torch.tensor([self._adv_returns[t.transition_id][1] for t in transitions], dtype=torch.float32, device=self.device)
        state_batch = self._collate_states(states)
        batch = {
            'states': state_batch,
            'actions': actions,
            'old_log_probs': old_log_probs,
            'old_values': old_values,
            'advantages': advantages,
            'returns': returns
        }
        return batch

    def _collate_states(self, states: List[State]):
        """Collate States assuming consistent presence of ctx, action_features, and action_mask.

        Returns dict with 'map', 'ctx', 'action_features', 'action_mask'.
        """
        if len(states) == 0:
            raise ValueError("_collate_states received empty state list")

        maps_np = [s.map for s in states]
        ctx_all = [s.ctx for s in states]
        if any(c is None for c in ctx_all):
            raise ValueError("All states must have ctx in new design")

    # Reconstruct or validate per-action features + action masks.
        action_feats_list = []
        action_masks_list = []
        # Attempt to import model constants for validation; fall back to defaults.
        try:
            from snake_sim.rl.models.ppo_model import SnakePPONet  # local import to avoid circular issues
            exp_actions = SnakePPONet.NUM_ACTIONS
            exp_feat_dim = SnakePPONet.ACTION_FEAT_DIM
        except Exception:
            exp_actions, exp_feat_dim = 4, 3

        for s in states:
            meta = s.meta
            af = meta['action_features']
            # Validate shape
            af = np.asarray(af)
            if af.ndim != 2 or af.shape[0] != exp_actions or af.shape[1] != exp_feat_dim:
                raise ValueError(f"action_features shape {af.shape} mismatch; expected ({exp_actions},{exp_feat_dim})")
            action_feats_list.append(af)

            if 'action_mask' not in meta:
                raise KeyError("State meta missing action_mask; fix state builder/adapters")
            am = np.asarray(meta['action_mask'])
            if am.ndim != 1 or am.shape[0] != exp_actions:
                raise ValueError(f"action_mask shape {am.shape} mismatch; expected ({exp_actions},)")
            action_masks_list.append(am.astype(np.float32))

        # Use non-blocking transfers for better GPU utilization
        map_batch = torch.from_numpy(np.stack(maps_np)).float().to(self.device, non_blocking=True)
        ctx_batch = torch.from_numpy(np.stack(ctx_all)).float().to(self.device, non_blocking=True)
        af_batch = torch.from_numpy(np.stack(action_feats_list)).float().to(self.device, non_blocking=True)
        am_batch = torch.from_numpy(np.stack(action_masks_list)).to(self.device, non_blocking=True)
        # keep float32 in batch; cast to bool right before masking logits
        return {'map': map_batch, 'ctx': ctx_batch, 'action_features': af_batch, 'action_mask': am_batch.float()}
    # ---- Forward & Update ----
    def _forward(self, state_batch):
        logits, value = self.model(state_batch)
        return logits, value.squeeze(-1)

    def _ppo_update(self, batch: Dict, epoch: int = 0):
        """Single minibatch PPO update.
        
        Args:
            batch: Dictionary containing states, actions, advantages, returns, etc.
            epoch: Current training epoch (for logging)
        """
        self.model.train()

        states = batch['states']  # dict of batched tensors
        actions = batch['actions']
        old_log_probs = batch['old_log_probs']
        old_values = batch.get('old_values', None)
        advantages = batch['advantages']
        returns = batch['returns']
        batch_size = len(actions)

        # Note: advantages are already normalized in _compute_gae()
        # Do NOT renormalize here - it causes double normalization artifacts
        # advantages already have mean≈0, std≈1

        # Normalize returns (critic target) if enabled
        if self._enable_value_norm:
            returns_np = returns.cpu().numpy()
            self._update_return_stats(returns_np)
            normalized_returns = (returns - self._return_mean) / (self._return_std + 1e-8)
        else:
            normalized_returns = returns

        # Shuffle once per update for decorrelation
        indices = torch.randperm(batch_size, device=self.device)
        shuffled_states = {
            'map': states['map'][indices],
            'ctx': states['ctx'][indices],
            'action_features': states['action_features'][indices],
            'action_mask': states['action_mask'][indices],
        }
        shuffled_actions = actions[indices]
        shuffled_old_log_probs = old_log_probs[indices]
        shuffled_old_values = old_values[indices] if old_values is not None else None
        shuffled_advantages = advantages[indices]
        shuffled_returns = normalized_returns[indices]

        # Forward with autocast (AMP) if available
        try:
            from torch import amp as _torch_amp
            autocast_ctx = _torch_amp.autocast('cuda', enabled=(self._amp_enabled and self.cfg.use_amp and not self._amp_runtime_disabled))
        except Exception:
            autocast_ctx = torch.cuda.amp.autocast(enabled=(self._amp_enabled and self.cfg.use_amp and not self._amp_runtime_disabled))
        with autocast_ctx:
            logits_raw, values = self._forward(shuffled_states)
        logits_raw = logits_raw.float()
        values = values.float()

        # ---- Numerical safety checks ----
        # If forward produced NaNs/Infs, bail before creating distributions.
        # This prevents publishing a broken checkpoint that crashes actor processes.
        if (not torch.isfinite(logits_raw).all()) or (not torch.isfinite(values).all()):
            log.error(
                "🚨 Non-finite model outputs in PPO update %s (epoch=%s). Skipping update.",
                self._update_counter,
                epoch,
            )
            return None

        # Hard action masking (recommended for snake).
        action_mask = shuffled_states['action_mask'].to(dtype=torch.bool)
        logits = apply_action_mask_to_logits(logits_raw, action_mask)

        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(shuffled_actions).float()
        entropy = dist.entropy().mean().float()

        with torch.no_grad():
            kl_div = (shuffled_old_log_probs - log_probs).mean().item()

        ratio = torch.exp(log_probs - shuffled_old_log_probs)
        surr1 = ratio * shuffled_advantages
        surr2 = torch.clamp(ratio, 1 - self.cfg.clip_range, 1 + self.cfg.clip_range) * shuffled_advantages
        policy_loss = -torch.mean(torch.min(surr1, surr2))

        # PPO diagnostics (computed on the same minibatch)
        with torch.no_grad():
            clipfrac = float(((ratio - 1.0).abs() > self.cfg.clip_range).float().mean().item())
            ratio_mean = float(ratio.mean().item())
            ratio_std = float(ratio.std(unbiased=False).item())
            # Standard PPO approx KL: mean(old_logp - new_logp)
            approx_kl = float((shuffled_old_log_probs - log_probs).mean().item())

        # If KL blows up, it's very likely we took an oversized update.
        # Skipping protects against publishing a broken checkpoint.
        if approx_kl > float(self.cfg.kl_hard_limit):
            log.error(
                "🚨 KL hard limit exceeded (approx_kl=%.4f > %.4f). Skipping update.",
                approx_kl,
                float(self.cfg.kl_hard_limit),
            )
            return None

        # PPO-style value clipping: stabilize critic by limiting deviation from OLD (rollout-time) values.
        # NOTE: Using values.detach() here is a bug (it makes the clip delta always 0). We must use
        # the stored rollout-time estimates.
        if shuffled_old_values is None:
            # Safety fallback (shouldn't happen in normal training): behave like unclipped MSE.
            values_clipped = values
        else:
            values_old = shuffled_old_values.detach()
            values_clipped = values_old + torch.clamp(values - values_old, -self.cfg.vf_clip_range, self.cfg.vf_clip_range)
        value_loss_unclipped = F.mse_loss(values, shuffled_returns)
        value_loss_clipped = F.mse_loss(values_clipped, shuffled_returns)
        value_loss = torch.max(value_loss_unclipped, value_loss_clipped)

        # ---- Auxiliary legality loss (pre-mask logits) ----
        # Goal: reduce invalid_action_frac by teaching the network that illegal moves
        # should have low logits even before masking.
        # We only use it when there exists at least one invalid action in the minibatch.
        illegal_logit_loss = logits_raw.new_tensor(0.0)
        if self.cfg.illegal_logit_coef > 0:
            invalid_mask = ~action_mask
            if invalid_mask.any():
                illegal_logits = logits_raw[invalid_mask]
                # softplus(x) ~ max(0, x) but smooth.
                illegal_logit_loss = F.softplus(illegal_logits - float(self.cfg.illegal_logit_margin)).mean()

        # Critic quality diagnostic: explained variance of returns by value predictions
        with torch.no_grad():
            v_pred = values
            v_y = shuffled_returns
            v_y_var = torch.var(v_y)
            if float(v_y_var.item()) < 1e-12:
                explained_variance = float('nan')
            else:
                explained_variance = float((1.0 - torch.var(v_y - v_pred) / v_y_var).item())

        # Value loss can be moderately large without implying divergence; avoid skipping too eagerly
        # or you end up freezing learning whenever returns get noisy.
        if value_loss.item() > 100.0:
            log.error(f"🚨 VALUE FUNCTION EXPLOSION: loss={value_loss.item():.2f}")
            log.error(f"   Value predictions - min: {values.min().item():.2f}, max: {values.max().item():.2f}, mean: {values.mean().item():.2f}")
            log.error(f"   Return targets - min: {shuffled_returns.min().item():.2f}, max: {shuffled_returns.max().item():.2f}, mean: {shuffled_returns.mean().item():.2f}")
            log.error("   Skipping this update to prevent collapse")
            return None

        total_loss = (
            policy_loss
            + self.cfg.value_coef * value_loss
            - self.cfg.entropy_coef * entropy
            + self.cfg.illegal_logit_coef * illegal_logit_loss
        )

        # Guard against NaN/Inf loss (can happen with extreme ratios/advantages).
        if not torch.isfinite(total_loss):
            log.error("🚨 Non-finite total_loss detected; skipping update.")
            return None

        try:
            self.optimizer.zero_grad(set_to_none=True)
            if self._amp_enabled and self.cfg.use_amp and not self._amp_runtime_disabled:
                self._scaler.scale(total_loss).backward()
                self._scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self._scaler.step(self.optimizer)
                self._scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()
        except RuntimeError as e:
            if 'expected Half' in str(e) and not self._amp_runtime_disabled:
                log.warning(f"AMP dtype mismatch detected; disabling AMP: {e}")
                self._amp_runtime_disabled = True
                self.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()
            else:
                raise

        self._update_counter += 1
        current_return = float(returns.mean().item())
        self._check_stagnation_and_adapt(current_return)
        if self._snapshot_manager and (self._update_counter % self.cfg.save_every_updates == 0):
            self._save_snapshot()

        stats = {
            'policy_loss': float(policy_loss.item()),
            'value_loss': float(value_loss.item()),
            'entropy': float(entropy.item()),
            'kl_div': float(kl_div),
            'update': self._update_counter,
            'advantages_mean': float(advantages.mean().item()),
            'returns_mean': float(returns.mean().item()),
            'batch_size': batch_size,
            'epochs_completed': self.cfg.train_epochs,  # For CSV compatibility
            'entropy_coef': self.cfg.entropy_coef,
            'exploration_boosted': self._exploration_boosted,
            # PPO diagnostics
            'clipfrac': clipfrac,
            'ratio_mean': ratio_mean,
            'ratio_std': ratio_std,
            'approx_kl': approx_kl,
            'explained_variance': explained_variance,
            # Mask diagnostic
            'invalid_action_frac': float((~action_mask).float().mean().item()),
            'illegal_logit_loss': float(illegal_logit_loss.item()) if illegal_logit_loss is not None else 0.0,
        }

        # --- Behavior metrics ---
        try:
            foods = int(getattr(self, '_last_batch_foods_eaten', 0))
            steps = int(batch_size)
            stats['foods_eaten'] = foods
            stats['foods_per_1k_steps'] = float(foods) / max(1, steps) * 1000.0
            stats['steps_per_food'] = float(steps) / max(1, foods)
        except Exception:
            pass

        # Add model diagnostics (needed for CSV; _log_detailed_metrics also adds these)
        try:
            stats.update(self._get_policy_mixer_stats())
        except Exception as e:
            log.debug(f"Mixer stats unavailable: {e}")
        # De-normalized value prediction logging for diagnostics
        try:
            if self._enable_value_norm:
                denorm_values = values * (self._return_std + 1e-8) + self._return_mean
            else:
                denorm_values = values
            value_pred_denorm_mean = float(denorm_values.mean().item())
            value_residual_mean = value_pred_denorm_mean - stats['returns_mean']
            stats['value_pred_denorm_mean'] = value_pred_denorm_mean
            stats['value_residual_mean'] = float(value_residual_mean)
        except Exception as e:
            log.debug(f"Value denorm logging failed: {e}")
        self._log_to_csv(stats)
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
            log.debug("Background PPO training thread started")
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
                               f"KL={stats.get('kl_div', 0):.4f}, "
                               f"Batch={stats['batch_size']}")
                        
                        # Detect concerning patterns
                        if stats.get('kl_div', 0) > self.cfg.kl_target:
                            log.warning(f"⚠️  High KL divergence: {stats['kl_div']:.4f} - policy changing rapidly")
                        
                        if stats['value_loss'] > 10.0:
                            log.warning(f"⚠️  High value loss: {stats['value_loss']:.3f} - critic may be unstable")
                        
                        # Detailed stats less frequently 
                        if current_time - last_stats_time >= stats_interval:
                            queue_size = self._transition_queue.size()
                            log.info(f"📊 Training Overview (Update {stats['update']}):")
                            log.info(f"   Queue Size: {queue_size} | Batch Size: {stats['batch_size']}")
                            log.info(f"   Policy Loss: {stats['policy_loss']:.4f} | Value Loss: {stats['value_loss']:.4f}")
                            log.info(f"   Entropy: {stats['entropy']:.4f} | KL Div: {stats.get('kl_div', 0):.4f}")
                            log.info(f"   Avg Return: {stats['returns_mean']:.4f} | Best: {self._best_return:.4f}")
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

    def join_background(self, timeout: Optional[float] = None):
        """Join the background thread if running."""
        if self._bg_thread is not None:
            self._bg_thread.join(timeout=timeout)

    def _save_snapshot(self):
        """Save model snapshot using SnapshotManager."""
        if not self._snapshot_manager:
            return
        try:
            optimizers = {'optimizer': self.optimizer}
            path = self._snapshot_manager.save(
                step=self._update_counter,
                model=self.model,
                optimizers=optimizers
            )
            log.debug(f"Saved snapshot {path}")
            # Prune old snapshots (keep last 2)
            self._snapshot_manager.prune(keep_last=2)
        except Exception as e:
            log.warning(f"Failed saving snapshot: {e}")
    
    def close(self):
        """Cleanup resources including CSV file."""
        if self._csv_file:
            try:
                if not self._csv_file.closed:
                    self._csv_file.flush()
                    self._csv_file.close()
                log.debug("📝 Training statistics CSV file closed")
            except Exception as e:
                log.warning(f"Failed to close CSV file: {e}")
            finally:
                # Prevent double-close attempts (e.g., explicit close() then __del__).
                self._csv_writer = None
                self._csv_file = None
        
        # Stop background thread if running
        if self._bg_running:
            self.stop_background(join=True, timeout=5.0)
    
    def __del__(self):
        """Ensure CSV file is closed on deletion."""
        self.close()
