from __future__ import annotations

from typing import List, Tuple, Optional

from snake_sim.environment.types import Coord, StrategyConfig, EnvMetaData
from snake_sim.environment.interfaces.snake_strategy_interface import ISnakeStrategy

import snake_sim.debugging as debug
from typing import List, Tuple, Optional, Dict, Deque
from dataclasses import dataclass
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from snake_sim.environment.types import Coord
from snake_sim.environment.interfaces.snake_strategy_interface import ISnakeStrategy

import snake_sim.debugging as debug
from snake_sim.cpp_bindings.area_check import AreaChecker
from snake_sim.cpp_bindings.utils import get_visitable_tiles


debug.activate_debug()
# debug.enable_debug_for_all()
debug.enable_debug_for("_save_model")


@dataclass
class _OptionData:
    coord: Optional[Coord]
    area_result: Dict
    features: np.ndarray  # per-option feature vector
    valid: bool
    score: float = 0.0


class _ReplayBuffer:
    def __init__(self, capacity: int):
        self._capacity = capacity
        self._data: List[Dict] = []
        self._idx = 0

    def push(self, item: Dict):
        if len(self._data) < self._capacity:
            self._data.append(item)
        else:
            self._data[self._idx] = item
        self._idx = (self._idx + 1) % self._capacity

    def sample(self, batch_size: int) -> List[Dict]:
        return random.sample(self._data, min(batch_size, len(self._data)))

    def __len__(self):
        return len(self._data)


class _DeepStrategyNet(nn.Module):
    """Shared trunk + per-option head. Lightweight initial version."""
    def __init__(self, in_channels: int, g_dim: int, o_dim: int, ctx_dim: int = 64, head_hidden: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.global_proj = nn.Linear(g_dim, ctx_dim)
        self.ctx_proj = nn.Linear(64, ctx_dim)
        self.head1 = nn.Linear(ctx_dim + o_dim, head_hidden)
        self.head2 = nn.Linear(head_hidden, 1)

    def forward(self, map_tensor: torch.Tensor, global_feats: torch.Tensor, option_feats: torch.Tensor) -> torch.Tensor:
        # map_tensor: (B,C,H,W), global_feats: (B,g_dim), option_feats: (B,N,o_dim)
        x = F.relu(self.conv1(map_tensor))
        x = F.relu(self.conv2(x))
        x = torch.mean(x, dim=(2, 3))  # global avg pool -> (B,64)
        ctx = F.relu(self.ctx_proj(x)) + F.relu(self.global_proj(global_feats))  # (B,ctx_dim)
        B, N, o_dim = option_feats.shape
        ctx_expanded = ctx.unsqueeze(1).expand(-1, N, -1)
        z = torch.cat([ctx_expanded, option_feats], dim=-1)  # (B,N,ctx_dim+o_dim)
        h = F.relu(self.head1(z))
        out = self.head2(h).squeeze(-1)  # (B,N)
        return out


class DeepStrategy(ISnakeStrategy):
    """DeepStrategy with training/inference toggle.

    Current implementation:
      - Builds simple map tensor channels.
      - Extracts per-option features from AreaChecker + distances.
      - If training: epsilon-greedy, replay buffer, periodic optimization (DQN-style).
      - If inference: greedy selection; falls back to heuristic if model unavailable.
    """

    DEFAULT_CONFIG = {
        'training': False,
        'epsilon_start': 0.15,
        'epsilon_min': 0.05,
        'epsilon_decay_steps': 20000,
        'gamma': 0.95,
        'lr': 1e-3,
        'replay_size': 20000,
        'batch_size': 64,
        'train_interval': 20,
        'warmup_steps': 500,
        'save_interval': 5000,
        'model_path': 'models/deep_strategy.pt',
        'device': 'auto'  # 'auto' | 'cpu' | 'cuda'
    }

    def __init__(self, config: Dict = None):
        # Base strategy has no config parameter; we store config locally.
        super().__init__(config)
        self._area_checker: Optional[AreaChecker] = None
        self.config = {**DeepStrategy.DEFAULT_CONFIG, **(config or {})}
        self.training_mode = bool(self.config['training'])

        # Device selection
        device_cfg = self.config.get('device', 'auto')
        if device_cfg == 'auto':
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            if device_cfg == 'cuda' and not torch.cuda.is_available():
                self._device = torch.device('cpu')
            else:
                self._device = torch.device(device_cfg)
        debug.debug_print(f"DeepStrategy: using device {self._device}")

        # RL state
        self._policy_net: Optional[_DeepStrategyNet] = None
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._replay: Optional[_ReplayBuffer] = None
        self._step_counter = 0
        self._epsilon = self.config['epsilon_start'] if self.training_mode else 0.0
        self._pending_transition: Optional[Dict] = None  # holds previous state/action until next state arrives

        # Lazily initialized after first map seen (need dimensions)
        self._initialized_model = False

    # ---- Strategy Interface ----
    def get_wanted_tile(self) -> Optional[Coord]:
        if self._snake is None:
            return None
        if self._area_checker is None:
            self._init_area_checker()

        env_meta = self._snake.get_env_meta_data()
        s_map = self._snake.get_map()
        head = self._snake.get_head_coord()

        traversable = self._visitable_tiles()
        if not traversable:
            debug.debug_print("DeepStrategy: no traversable candidates")
            return None

        # Build current state representations
        map_tensor = self._build_map_tensor(env_meta, s_map, head)
        global_feats = self._build_global_features(env_meta, s_map, head)
        options = self._build_options(traversable, env_meta, s_map, head)

        # If we have a pending previous transition, finalize its reward and store
        if self.training_mode and self._pending_transition is not None:
            self._finalize_previous_transition(map_tensor, global_feats, options)

        # Initialize model if first time
        if not self._initialized_model:
            # map_tensor shape = (C,H,W) so channels dimension is index 0
            self._init_model(map_tensor.shape[0], global_feats.shape[0], options[0].features.shape[0])

        # Prepare tensors for forward
        option_feats_tensor = torch.from_numpy(np.stack([o.features for o in options])).float().unsqueeze(0).to(self._device)
        map_tensor_t = torch.from_numpy(map_tensor).float().unsqueeze(0).to(self._device)
        global_feats_t = torch.from_numpy(global_feats).float().unsqueeze(0).to(self._device)

        scores = self._policy_net(map_tensor_t, global_feats_t, option_feats_tensor).detach().cpu().numpy()[0]

        # Combine scores with simple heuristic fallback if model weak (early) or for added shaping
        for i, o in enumerate(options):
            if not self.training_mode and self._step_counter < 10:
                # Use heuristic early just to avoid random initial moves
                scores[i] += self._heuristic_bonus(o)
            o.score = float(scores[i])

        # Action selection
        action_index = self._select_action(options)
        chosen = options[action_index]

        # Store pending transition (state before executing chosen action)
        valid_mask = np.array([1 if o.valid else 0 for o in options], dtype=np.float32)
        if self.training_mode:
            self._pending_transition = {
                'state_map': map_tensor,
                'global_feats': global_feats,
                'option_feats': np.stack([o.features for o in options]),
                'action_index': action_index,
                'chosen_coord': chosen.coord,
                'min_food_dist': self._min_food_distance(env_meta, s_map, head),
                'valid_mask': valid_mask
            }

        self._step_counter += 1
        self._update_epsilon()

        debug.debug_print(f"DeepStrategy step={self._step_counter} epsilon={self._epsilon:.3f} chosen={chosen.coord} score={chosen.score:.2f}")
        return chosen.coord

    # ---- Model / Training Helpers ----
    def _init_model(self, in_channels: int, g_dim: int, o_dim: int):
        self._policy_net = _DeepStrategyNet(in_channels, g_dim, o_dim).to(self._device)
        if self.training_mode:
            self._optimizer = torch.optim.Adam(self._policy_net.parameters(), lr=self.config['lr'])
            self._replay = _ReplayBuffer(self.config['replay_size'])
        else:
            # Try loading weights if file exists
            path = self.config['model_path']
            try:
                state = torch.load(path, map_location=self._device)
                self._policy_net.load_state_dict(state['policy_state'])
                debug.debug_print(f"DeepStrategy: loaded model from {path}")
            except Exception as e:
                debug.debug_print(f"DeepStrategy: no model loaded ({e}) - using random weights")
        self._initialized_model = True

    def _finalize_previous_transition(self, next_map_tensor, next_global_feats, next_options: List[_OptionData]):
        prev = self._pending_transition
        self._pending_transition = None
        # Reward based on food progress & margin improvement
        reward = self._compute_reward(prev, next_map_tensor, next_global_feats)
        next_valid_mask = np.array([1 if o.valid else 0 for o in next_options], dtype=np.float32)
        experience = {
            'state_map': prev['state_map'],
            'global_feats': prev['global_feats'],
            'option_feats': prev['option_feats'],
            'valid_mask': prev['valid_mask'],
            'action_index': prev['action_index'],
            'reward': reward,
            'next_state_map': next_map_tensor,
            'next_global_feats': next_global_feats,
            'next_option_feats': np.stack([o.features for o in next_options]),
            'next_valid_mask': next_valid_mask,
            'done': False  # terminal handling TBD
        }
        self._replay.push(experience)
        if len(self._replay) >= self.config['warmup_steps'] and self._step_counter % self.config['train_interval'] == 0:
            self._train_step()
        if self._step_counter % self.config['save_interval'] == 0:
            self._save_model()

    def _train_step(self):
        batch = self._replay.sample(self.config['batch_size'])
        if len(batch) == 0:
            return
        # Assemble tensors
        state_maps = torch.from_numpy(np.stack([b['state_map'] for b in batch])).float().to(self._device)
        global_feats = torch.from_numpy(np.stack([b['global_feats'] for b in batch])).float().to(self._device)
        option_feats = torch.from_numpy(np.stack([b['option_feats'] for b in batch])).float().to(self._device)
        valid_mask = torch.from_numpy(np.stack([b['valid_mask'] for b in batch])).float().to(self._device)  # (B,4)
        actions = torch.tensor([b['action_index'] for b in batch], dtype=torch.long).to(self._device)  # (B,)
        rewards = torch.tensor([b['reward'] for b in batch], dtype=torch.float32).to(self._device)
        next_maps = torch.from_numpy(np.stack([b['next_state_map'] for b in batch])).float().to(self._device)
        next_globals = torch.from_numpy(np.stack([b['next_global_feats'] for b in batch])).float().to(self._device)
        next_options = torch.from_numpy(np.stack([b['next_option_feats'] for b in batch])).float().to(self._device)
        next_valid_mask = torch.from_numpy(np.stack([b['next_valid_mask'] for b in batch])).float().to(self._device)  # (B,4)
        dones = torch.tensor([b['done'] for b in batch], dtype=torch.float32).to(self._device)

        # Current Q (mask invalid before gather for numerical consistency)
        q_values_all = self._policy_net(state_maps, global_feats, option_feats)  # (B,4)
        q_values = q_values_all * valid_mask + (1 - valid_mask) * (-1e6)
        q_chosen = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Next Q (greedy policy net) with masking
        with torch.no_grad():
            next_q_values_all = self._policy_net(next_maps, next_globals, next_options)  # (B,4)
            next_q_values = next_q_values_all * next_valid_mask + (1 - next_valid_mask) * (-1e6)
            next_q_max = torch.max(next_q_values, dim=1).values
            targets = rewards + (1 - dones) * self.config['gamma'] * next_q_max

        loss = F.smooth_l1_loss(q_chosen, targets)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        debug.debug_print(f"DeepStrategy train step loss={float(loss):.4f} buffer={len(self._replay)}")

    def _save_model(self):
        path = self.config['model_path']
        debug.debug_print(f"DeepStrategy: saving model to {path}")
        try:
            import os
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save({'policy_state': self._policy_net.state_dict(), 'step': self._step_counter, 'config': self.config}, path)
            debug.debug_print(f"DeepStrategy: model saved to {path}")
        except Exception as e:
            debug.debug_print(f"DeepStrategy: failed saving model ({e})")

    def _update_epsilon(self):
        if not self.training_mode:
            return
        decay_steps = self.config['epsilon_decay_steps']
        min_eps = self.config['epsilon_min']
        start = self.config['epsilon_start']
        progress = min(1.0, self._step_counter / decay_steps)
        self._epsilon = max(min_eps, start - (start - min_eps) * progress)

    # ---- Feature / Option Construction ----
    def _build_map_tensor(self, env_meta: EnvMetaData, s_map: np.ndarray, head: Coord) -> np.ndarray:
        # Channels: my_head, my_body, food, opponents_heads, opponents_bodies
        C = 5
        tensor = np.zeros((C, env_meta.height, env_meta.width), dtype=np.float32)
        tensor[0, head.y, head.x] = 1.0
        body_coords = list(self._snake.get_body_coords())
        # body (exclude head)
        for b in body_coords[1:]:
            tensor[1, b.y, b.x] = 1.0
        # food
        food_val = env_meta.food_value
        tensor[2][s_map == food_val] = 1.0
        # opponents
        my_head_val, my_body_val = self._snake.get_self_map_values()
        opponent_head_vals = []
        opponent_body_vals = []
        for sid, vals in env_meta.snake_values.items():
            hv = vals.get('head_value')
            bv = vals.get('body_value')
            if hv == my_head_val and bv == my_body_val:
                continue
            opponent_head_vals.append(hv)
            opponent_body_vals.append(bv)
        if opponent_head_vals:
            mask_heads = np.isin(s_map, opponent_head_vals)
            tensor[3][mask_heads] = 1.0
        if opponent_body_vals:
            mask_bodies = np.isin(s_map, opponent_body_vals)
            tensor[4][mask_bodies] = 1.0
        return tensor

    def _build_global_features(self, env_meta: EnvMetaData, s_map: np.ndarray, head: Coord) -> np.ndarray:
        body_len = getattr(self._snake, '_length', len(self._snake.get_body_coords()))
        max_len_cap = env_meta.width * env_meta.height / 2.0
        body_len_norm = min(1.0, body_len / max_len_cap)
        min_food_dist = self._min_food_distance(env_meta, s_map, head)
        min_food_dist_norm = min_food_dist / (env_meta.width + env_meta.height) if min_food_dist >= 0 else 1.0
        total_food = int(np.sum(s_map == env_meta.food_value))
        total_food_norm = min(1.0, total_food / 50.0)
        forward_margin_norm = 0.0  # will be derived by checking forward tile
        # simple forward tile area check
        forward_tile = self._forward_tile(head, env_meta.width, env_meta.height)
        if forward_tile is not None:
            res = self._area_check_wrapper(forward_tile)
            if res:
                margin = res.get('margin', 0) or 0
                forward_margin_norm = margin / (body_len + 1)
        return np.array([body_len_norm, min_food_dist_norm, total_food_norm, forward_margin_norm], dtype=np.float32)

    def _forward_tile(self, head: Coord, width: int, height: int) -> Optional[Coord]:
        # simplistic: up
        nx, ny = head.x, head.y - 1
        if 0 <= nx < width and 0 <= ny < height:
            return Coord(nx, ny)
        return None

    def _build_options(self, traversable: List[Coord], env_meta: EnvMetaData, s_map: np.ndarray, head: Coord) -> List[_OptionData]:
        # Always return exactly 4 options in fixed direction order: Up, Right, Down, Left
        dirs = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        dir_coords = [Coord(head.x + dx, head.y + dy) for dx, dy in dirs]
        body_len = getattr(self._snake, '_length', len(self._snake.get_body_coords()))
        min_food_current = self._min_food_distance(env_meta, s_map, head)

        # Gather area results only for valid traversable coords
        area_results: Dict[Coord, Dict] = {}
        best_margin = 0
        for tile in dir_coords:
            if (0 <= tile.x < env_meta.width and 0 <= tile.y < env_meta.height and
                s_map[tile.y, tile.x] in (env_meta.free_value, env_meta.food_value)):
                res = self._area_check_wrapper(tile)
                area_results[tile] = res
                best_margin = max(best_margin, (res.get('margin', 0) or 0))

        options: List[_OptionData] = []
        for tile, (dx, dy) in zip(dir_coords, dirs):
            if tile in area_results:
                res = area_results[tile]
                margin = res.get('margin', 0) or 0
                food_count = res.get('food_count', 0) or 0
                tile_count = res.get('tile_count', 0) or 0
                margin_norm = margin / (body_len + 1)
                relative_margin = margin / (best_margin + 1e-6) if best_margin > 0 else 0.0
                food_frac_area = food_count / (tile_count + 1e-6)
                min_food_after = self._min_food_distance(env_meta, s_map, tile)
                food_dist_delta = (min_food_current - min_food_after) if (min_food_current >= 0 and min_food_after >= 0) else 0.0
                food_dist_delta_norm = food_dist_delta / (env_meta.width + env_meta.height)
                immediate_food_bonus = 1.0 if s_map[tile.y, tile.x] == env_meta.food_value else 0.0
                dir_one_hot = self._dir_one_hot((dx, dy))
                feats = np.array([
                    margin_norm,
                    relative_margin,
                    food_frac_area,
                    food_dist_delta_norm,
                    immediate_food_bonus,
                    *dir_one_hot
                ], dtype=np.float32)
                options.append(_OptionData(coord=tile, area_result=res, features=feats, valid=True))
            else:
                # Pad invalid option (not traversable / out of bounds)
                dir_one_hot = self._dir_one_hot((dx, dy))
                feats = np.array([
                    0.0,  # margin_norm
                    0.0,  # relative_margin
                    0.0,  # food_frac_area
                    0.0,  # food_dist_delta_norm
                    0.0,  # immediate_food_bonus
                    *dir_one_hot
                ], dtype=np.float32)
                options.append(_OptionData(coord=None, area_result={}, features=feats, valid=False))
        return options

    def _dir_one_hot(self, dir_vec: Tuple[int, int]) -> List[float]:
        mapping = {(0, -1): 0, (1, 0): 1, (0, 1): 2, (-1, 0): 3}
        idx = mapping.get(dir_vec, -1)
        one_hot = [0.0, 0.0, 0.0, 0.0]
        if idx >= 0:
            one_hot[idx] = 1.0
        return one_hot

    def _select_action(self, options: List[_OptionData]) -> int:
        scores = [o.score if o.valid else -1e9 for o in options]
        valid_indices = [i for i, o in enumerate(options) if o.valid]
        if not valid_indices:
            return 0  # fallback
        if self.training_mode and random.random() < self._epsilon:
            return random.choice(valid_indices)
        best_idx = max(valid_indices, key=lambda i: scores[i])
        return best_idx

    def _heuristic_bonus(self, option: _OptionData) -> float:
        # reuse earlier heuristic lightly to guide early steps
        res = option.area_result
        margin = res.get('margin', 0) or 0
        food_count = res.get('food_count', 0) or 0
        return 0.5 * food_count + 0.1 * margin

    def _compute_reward(self, prev: Dict, next_map_tensor, next_global_feats) -> float:
        # Basic shaping: food progress & margin improvement
        prev_min_food = prev['min_food_dist']
        env_meta = self._snake.get_env_meta_data()
        s_map = self._snake.get_map()
        head = self._snake.get_head_coord()
        new_min_food = self._min_food_distance(env_meta, s_map, head)
        food_progress = (prev_min_food - new_min_food) if (prev_min_food >= 0 and new_min_food >= 0) else 0.0
        food_reward = 5.0 if self._ate_food(prev['chosen_coord'], env_meta, s_map) else 0.0
        margin_improvement = next_global_feats[3]  # forward_margin_norm of new state
        reward = food_reward + 2.0 * food_progress + 0.5 * margin_improvement
        return float(reward)

    def _ate_food(self, coord: Coord, env_meta, s_map: np.ndarray) -> bool:
        return s_map[coord.y, coord.x] == env_meta.food_value

    def _min_food_distance(self, env_meta, s_map: np.ndarray, from_coord: Coord) -> int:
        food_positions = np.argwhere(s_map == env_meta.food_value)
        if food_positions.size == 0:
            return -1
        dists = [abs(from_coord.x - x) + abs(from_coord.y - y) for y, x in food_positions]
        return min(dists) if dists else -1

    # --- Existing helpers retained / adapted ---
    def _init_area_checker(self):
        env_meta = self._snake.get_env_meta_data()
        head_value, body_value = self._snake.get_self_map_values()
        self._area_checker = AreaChecker(
            env_meta.food_value,
            env_meta.free_value,
            body_value,
            head_value,
            env_meta.width,
            env_meta.height
        )
        debug.debug_print("DeepStrategy: AreaChecker initialized")

    def _visitable_tiles(self) -> List[Coord]:
        env_meta = self._snake.get_env_meta_data()
        s_map = self._snake.get_map()
        head = self._snake.get_head_coord()

        visitable_tiles = get_visitable_tiles(
            s_map,
            env_meta.width,
            env_meta.height,
            head,
            [env_meta.free_value, env_meta.food_value]
        )
        return list(map(lambda c: Coord(*c), visitable_tiles))

    def _area_check_wrapper(self, start_coord: Coord):
        if self._area_checker is None:
            return {}
        s_map_copy = self._snake.get_map().copy()
        body_coords = list(self._snake.get_body_coords())
        result = self._area_checker.area_check(
            s_map_copy,
            body_coords,
            start_coord,
            target_margin=0,
            food_check=True,
            complete_area=False,
            exhaustive=False
        )
        return result