
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Sequence, Protocol, Optional

import math
import numpy as np
import sys


from snake_sim.cpp_bindings.utils import get_dir_to_tile, get_visitable_tiles, distance_to_tile_with_value, dist_heat_map
from snake_sim.cpp_bindings.area_check import AreaChecker
from snake_sim.map_utils.general import print_map
from snake_sim.environment.types import EnvMetaData, EnvStepData, Coord, AreaCheckResult
import snake_sim.rl.constants as consts
from snake_sim.rl.types import State

BASE_STATE_VERSION = 1


ADAPTER_CACHE = {}  # global cache for adapters if needed


def print_channel(channel_map: np.ndarray, free_value=0, food_value=0, blocked_value=0, head_value=0, body_value=0):
    print_map(
        s_map=channel_map,
        free_value=free_value,
        food_value=food_value,
        blocked_value=blocked_value,
        head_value=head_value,
        body_value=body_value
    )

def print_state(state: State):
    print("State map shape:", state.map.shape)
    print_channel(state.map[0], head_value=1)
    print_channel(state.map[1], body_value=1)
    print_channel(state.map[2], food_value=1)
    print_channel(state.map[3], blocked_value=1)
    _print_red_heatmap(state.map[4])
    print("State context:", state.ctx)
    print("State meta:", state.meta)
    sys.stdout.flush()


def _print_red_heatmap(norm_map: np.ndarray) -> None:
    """Print a normalized (H,W) map as colored blocks.

    Each cell is printed as two spaces with an ANSI 24-bit background color.
      - value 0.0 => bright red
      - value 1.0 => black
    """
    if norm_map is None or not hasattr(norm_map, 'shape') or len(norm_map.shape) != 2:
        print("<heatmap: invalid shape>")
        return

    v = np.clip(np.asarray(norm_map, dtype=np.float32), 0.0, 1.0)
    for y in range(v.shape[0]):
        parts: List[str] = []
        for x in range(v.shape[1]):
            red = int(round((1.0 - float(v[y, x])) * 255.0))
            parts.append(f"\x1b[48;2;{red};0;0m  \x1b[0m")
        print("".join(parts))


def _get_visitable_tiles(
        s_map: np.ndarray,
        env_meta: EnvMetaData,
        head_coord: Coord
    ) -> List[Coord]:
    if 'visitable_tiles' in ADAPTER_CACHE:
        return ADAPTER_CACHE['visitable_tiles']
    tile_tuples = get_visitable_tiles(
        s_map,
        env_meta.width,
        env_meta.height,
        head_coord,
        [env_meta.free_value, env_meta.food_value]
    )
    tile_coords = [Coord(*t) for t in tile_tuples]
    ADAPTER_CACHE['visitable_tiles'] = tile_coords
    return tile_coords


def _clean_dir_ctx():
    return np.zeros((len(consts.ACTION_ORDER),), dtype=np.float32)


@dataclass
class SnakeContext:
    snake_id: str
    head: Coord  # expected to be Coord-like with x,y
    body_coords: List[Coord]
    length: int


class IStateAdapter(Protocol):
    name: str
    def apply(self, state: State, step_data: EnvStepData, env_meta: EnvMetaData, snake_ctx: SnakeContext) -> None: ...


class BaseStateBuilder:
    def __init__(
        self,
        include_opponents: bool = True,
    ):
        self.include_opponents = include_opponents

    # New flexible build: per-channel buffers
    def build(self, env_meta: EnvMetaData, step_data: EnvStepData, snake_ctx: SnakeContext) -> State:
        channels_dict = self._build_channels(env_meta, step_data, snake_ctx)
        additional_channels = self._additional_channels(env_meta, step_data, snake_ctx)
        channels_dict.update(additional_channels)
        order = self._default_order(include_opponents=self.include_opponents)
        map_tensor = self._build_map(channels_dict, order)
        return State(
            map=map_tensor,
            ctx=np.array([min(1.0, snake_ctx.length / max(1, env_meta.width * env_meta.height))], dtype=np.float32),
            meta={
                'snake_id': snake_ctx.snake_id,
                'channel_order': order,
                'context_order': ['length_ratio'],
                'context_size': 1,
                'width': env_meta.width,
                'height': env_meta.height,
                'adapters': []
            }
        )

    def _create_food_dist_heat_map(self, env_meta: EnvMetaData, step_data: EnvStepData) -> np.ndarray:
        """Create a normalized food-distance channel.

        The underlying heat map returns integer-like distances (and may use -1 for unreachable).
        We convert it to float32 in [0, 1] by dividing by `food_dist_max` and clipping.

        Notes:
            - Distances <= 0 become 0.0 (including unreachable=-1).
            - Distances >= food_dist_max become 1.0.
        """
        heat_map_array = dist_heat_map(
            step_data.map,
            env_meta.width,
            env_meta.height,
            env_meta.free_value,
            env_meta.blocked_value,
            env_meta.food_value
        )

        # Convert to float32 and normalize into [0,1].
        distances = np.asarray(heat_map_array, dtype=np.float32)
        food_dist_max = (env_meta.width + env_meta.height) / 1.1

        # Treat unreachable or negative distances as 0 ("no signal").
        distances = np.maximum(distances, 0.0)
        norm = np.clip(distances / food_dist_max, 0.0, 1.0).astype(np.float32)

        # print("Food distance heatmap (red = close):")

        # # for row in norm:
        # #     print(" ".join([f"{val:4.2f}" for val in row]))

        # _print_red_heatmap(norm)

        return norm

    def _additional_channels(self, env_meta: EnvMetaData, step_data: EnvStepData, snake_ctx: SnakeContext) -> Dict[str, np.ndarray]:
        """Override to add more channels beyond the base ones."""
        return {
            "food_dist": self._create_food_dist_heat_map(env_meta, step_data)
        }
    
    def _build_channels(self, env_meta: EnvMetaData, step_data: EnvStepData, snake_ctx: SnakeContext) -> Dict[str, np.ndarray]:
        s_map = step_data.map
        H, W = env_meta.height, env_meta.width
        head = np.zeros((H, W), dtype=np.float32)
        body = np.zeros((H, W), dtype=np.float32)
        food = np.zeros((H, W), dtype=np.float32)
        blocked = np.zeros((H, W), dtype=np.float32)
        head[snake_ctx.head.y, snake_ctx.head.x] = 1.0
        for bc in snake_ctx.body_coords[1:]:
            body[bc.y, bc.x] = 1.0
        food[s_map == env_meta.food_value] = 1.0
        blocked_values = [env_meta.blocked_value] + [v for sid, vals in env_meta.snake_values.items() for v in vals.values()]
        blocked[np.isin(s_map, blocked_values)] = 1.0
        channels: Dict[str, np.ndarray] = {
            'head': head,
            'body': body,
            'food': food,
            'blocked': blocked,
        }
        if self.include_opponents:
            opp_heads = np.zeros((H, W), dtype=np.float32)
            opp_bodies = np.zeros((H, W), dtype=np.float32)
            opponent_head_vals: List[int] = []
            opponent_body_vals: List[int] = []
            for sid, vals in env_meta.snake_values.items():
                if sid == snake_ctx.snake_id or sid not in step_data.snakes or not step_data.snakes[sid].get('is_alive', False):
                    continue
                hv = vals.get('head_value')
                bv = vals.get('body_value')
                if hv is not None:
                    opponent_head_vals.append(hv)
                if bv is not None:
                    opponent_body_vals.append(bv)
            if opponent_head_vals:
                opp_heads[np.isin(s_map, opponent_head_vals)] = 1.0
            if opponent_body_vals:
                opp_bodies[np.isin(s_map, opponent_body_vals)] = 1.0
            channels['opp_heads'] = opp_heads
            channels['opp_bodies'] = opp_bodies
        return channels

    def _build_map(self, channels: Dict[str, np.ndarray], order: Sequence[str]) -> np.ndarray:
        """Stack channel dict into (C,H,W) array following provided order.

        Args:
            channels: mapping name -> (H,W) array.
            order: sequence of names defining stack order.
        Returns:
            np.ndarray shape (C,H,W) float32
        """
        stacked = np.stack([channels[name] for name in order], axis=0).astype(np.float32)
        return stacked

    def _default_order(self, include_opponents: bool = True) -> List[str]:
        base = ['head', 'body', 'food', 'blocked', 'food_dist']
        if include_opponents and self.include_opponents:
            base += ['opp_heads', 'opp_bodies']
        return base


class CompleteStateBuilder:
    def __init__(self, base_builder: BaseStateBuilder, adapters: List[IStateAdapter]=[]):
        self.base_builder = base_builder
        self.adapters = adapters
        
    def build(self, env_meta: EnvMetaData, step_data: EnvStepData, snake_ctx: SnakeContext) -> State:
        state = self.base_builder.build(env_meta, step_data, snake_ctx)
        ADAPTER_CACHE.clear()
        for adapter in self.adapters:
            adapter.apply(state, step_data, env_meta, snake_ctx)
        area_ctx = state.meta['area_ctx']  # shape (A,) - margin_frac for each action
        # Keep only margin_frac as action features. This gives spatial signal without giving away optimal actions.
        # Removed: safety_ctx (binary safe/unsafe) and food_ctx (direction to food) as they bypass learning.
        action_features = area_ctx.reshape(-1, 1).astype(np.float32)  # (A, 1)
        state.meta['action_features'] = action_features
        state.meta['action_feature_names'] = ['margin_frac']
        return state


class DirectionHintsAdapter:
    name = 'direction_hints'
    SAFE_MARGIN_FRAC = 0.06 # (margin / total_steps) >= SAFE_MARGIN_FRAC -> considered safe

    def __init__(self):
        self._area_checker = None

    def _init_area_checker(self, snake_ctx: SnakeContext, env_meta: EnvMetaData):
        snake_b_value = env_meta.snake_values[snake_ctx.snake_id]['body_value']
        snake_h_value = env_meta.snake_values[snake_ctx.snake_id]['head_value']
        self._area_checker = AreaChecker(
            env_meta.food_value,
            env_meta.free_value,
            snake_b_value,
            snake_h_value,
            env_meta.width,
            env_meta.height
        )

    def apply(self, state: State, step_data: EnvStepData, env_meta: EnvMetaData, snake_ctx: SnakeContext) -> None:
        # Initialize area checker lazily.
        if self._area_checker is None:
            self._init_area_checker(snake_ctx, env_meta)

        # Compute per-action area safety (margin fraction) and directional food hints.
        area_checks = self._get_area_checks(step_data.map, env_meta, step_data, snake_ctx)
        area_ctx = self._create_area_ctx(area_checks)  # (A,)
        safety_ctx = self._create_safety_ctx(area_checks)  # (A,)
        close_food_ctx = self._create_close_food_ctx(env_meta, step_data, snake_ctx)  # (A,)

        # Store them separately in meta (don't extend global ctx; keep it lean).
        state.meta['area_ctx'] = area_ctx.astype(np.float32)
        state.meta['safety_ctx'] = safety_ctx.astype(np.float32)
        state.meta['close_food_ctx'] = close_food_ctx.astype(np.float32)
        state.meta.setdefault('adapters', []).append(self.name)
        # For traceability we still record ordering labels but we do NOT bump context_size.
        state.meta.setdefault('context_order', ['length_ratio'])
        state.meta['area_ctx_labels'] = ['area_' + str(d_coord) for d_coord in consts.ACTION_ORDER]
        state.meta['safety_ctx_labels'] = ['safety_' + str(d_coord) for d_coord in consts.ACTION_ORDER]
        state.meta['close_food_ctx_labels'] = ['close_food_' + str(d_coord) for d_coord in consts.ACTION_ORDER]

    def _get_area_checks(self, s_map: np.ndarray, env_meta: EnvMetaData, step_data: EnvStepData, ctx: SnakeContext) -> Dict[Coord, AreaCheckResult]:
        head_coord = ctx.head
        visitable_tiles = _get_visitable_tiles(
            s_map, env_meta, head_coord
        )
        return {
            d_coord: self._get_area_check(d_coord, step_data, ctx) if head_coord + d_coord in visitable_tiles else None
            for d_coord in consts.ACTION_ORDER
        }
        
    def _get_area_check(self, coord: Coord, step_data: EnvStepData, snake_ctx: SnakeContext) -> AreaCheckResult:
        target_margin = max(10, math.ceil(self.SAFE_MARGIN_FRAC * len(snake_ctx.body_coords)))
        c_coord = snake_ctx.head + coord
        if not (area_check := ADAPTER_CACHE.get("area_checks", {}).get(coord)):
            result = self._area_checker.area_check(
                step_data.map,
                snake_ctx.body_coords,
                c_coord,
                target_margin=target_margin,
                food_check=False,
                complete_area=True,
                exhaustive=False
            )
            area_check = AreaCheckResult(**result)
            ADAPTER_CACHE.setdefault("area_checks", {})[coord] = area_check
        return area_check

    def _create_area_ctx(self, area_checks: dict[Coord, AreaCheckResult]) -> np.ndarray:
        ctx = _clean_dir_ctx()
        for coord, ac in area_checks.items():
            if ac is None:
                margin_frac = -1.0
            elif ac.has_tail:
                margin_frac = 1.0
            else:
                margin_frac = max(-1.0, min(1.0, ac.margin / max(1, ac.total_steps)))
            ctx[consts.ACTION_ORDER[coord]] = margin_frac
        return ctx

    def _create_safety_ctx(self, area_checks: dict[Coord, AreaCheckResult]) -> np.ndarray:
        ctx = _clean_dir_ctx()
        for coord, ac in area_checks.items():
            if ac is None:
                is_safe = 0.0
            else:
                is_safe = 1.0 if ac.margin / max(1, ac.total_steps) >= self.SAFE_MARGIN_FRAC else 0.0
            ctx[consts.ACTION_ORDER[coord]] = is_safe
        return ctx

    def _create_close_food_ctx(self, env_data: EnvMetaData, step_data: EnvStepData, snake_ctx: SnakeContext) -> np.ndarray:
        ctx = _clean_dir_ctx()
        distances = {}
        visitable_tiles = _get_visitable_tiles(
            step_data.map,
            env_data,
            snake_ctx.head
        )
        for tile in visitable_tiles:
            distance = distance_to_tile_with_value(
                step_data.map,
                env_data.width,
                env_data.height,
                tile,
                env_data.food_value,
                [env_data.free_value, env_data.food_value],
            )
            distances[tile] = distance
        dist_sum = sum([d for d in distances.values() if d >= 0])
        for c_dir, index in consts.ACTION_ORDER.items():
            distance = distances.get(snake_ctx.head + c_dir, -1)
            if dist_sum <= 0 or distance == -1:
                food_hint = 0.0
            else:
                food_hint = 0.5 + (0.5 - (distance / dist_sum) / 2)
            ctx[index] = food_hint

        return ctx


# ActionMaskAdapter removed: masking is disabled; policy learns safety from features


class ActionMaskAdapter:
    """Compute a hard action mask for the agent.

    The goal is to prevent *illegal* actions from being sampled/trained on.
    This is intentionally minimal and does NOT encode advanced safety heuristics
    like "will I die 3 steps later"; it only checks immediate legality.

    Stored in:
        state.meta['action_mask'] as np.ndarray shape (A,) float32 (1=valid, 0=invalid)
    """

    name = 'action_mask'

    def apply(self, state: State, step_data: EnvStepData, env_meta: EnvMetaData, snake_ctx: SnakeContext) -> None:
        s_map = step_data.map
        H, W = env_meta.height, env_meta.width
        head = snake_ctx.head

        mask = _clean_dir_ctx()
        for d_coord, idx in consts.ACTION_ORDER.items():
            nx, ny = head.x + d_coord.x, head.y + d_coord.y
            if nx < 0 or nx >= W or ny < 0 or ny >= H:
                mask[idx] = 0.0
                continue
            # Requested rule: treat anything > free_value as blocked.
            # This masks walls + other snake bodies/heads while keeping free/food tiles.
            mask[idx] = 0.0 if s_map[ny, nx] > env_meta.free_value else 1.0

        # Safety fallback: if everything is illegal (should not happen often),
        # allow all actions to avoid NaNs downstream; the env will resolve collisions.
        if mask.sum() == 0:
            mask[:] = 1.0

        state.meta['action_mask'] = mask.astype(np.float32)
        state.meta.setdefault('adapters', []).append(self.name)
