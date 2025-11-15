
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Sequence, Protocol, Optional

import math
import numpy as np


from snake_sim.cpp_bindings.utils import get_dir_to_tile, get_visitable_tiles
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
    print("State context:", state.ctx)
    print("State meta:", state.meta)



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
    def __init__(self, include_opponents: bool = True):
        self.include_opponents = include_opponents

    # New flexible build: per-channel buffers
    def build(self, env_meta: EnvMetaData, step_data: EnvStepData, snake_ctx: SnakeContext) -> State:
        channels_dict = self._build_channels(env_meta, step_data, snake_ctx)
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
        blocked[s_map == env_meta.blocked_value] = 1.0
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
                if sid == snake_ctx.snake_id:
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
        base = ['head', 'body', 'food', 'blocked']
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

        if self._area_checker is None:
            self._init_area_checker(snake_ctx, env_meta)

        area_checks = self._get_area_checks(step_data.map, env_meta, step_data, snake_ctx)
        area_ctx = self._create_area_ctx(area_checks)
        close_food_ctx = self._create_close_food_ctx(env_meta, step_data, snake_ctx)
        # Append to state context
        state.ctx = np.concatenate([state.ctx, area_ctx, close_food_ctx], axis=0)
        # Update meta
        state.meta['context_order'] += ['area_' + str(d_coord) for d_coord in consts.ACTION_ORDER]
        state.meta['context_order'] += ['close_food_' + str(d_coord) for d_coord in consts.ACTION_ORDER]
        state.meta['context_size'] += (area_ctx.shape[0] + close_food_ctx.shape[0])

    def _get_area_checks(self, s_map: np.ndarray, env_meta: EnvMetaData, step_data: EnvStepData, ctx: SnakeContext) -> Dict[Coord, AreaCheckResult]:
        head_coord = ctx.head
        visitable_tiles = get_visitable_tiles(
            s_map,
            env_meta.width,
            env_meta.height,
            head_coord,
            [env_meta.free_value, env_meta.food_value]
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
                complete_area=False,
                exhaustive=False
            )
            area_check = AreaCheckResult(**result)
            ADAPTER_CACHE.setdefault("area_checks", {})[coord] = area_check
        return area_check
    
    def _clean_dir_ctx(self):
        return np.zeros((len(consts.ACTION_ORDER),), dtype=np.float32)

    def _create_area_ctx(self, area_checks: dict[Coord, AreaCheckResult]) -> np.ndarray:
        ctx = self._clean_dir_ctx()
        for coord, ac in area_checks.items():
            if ac is None:
                margin_frac = -1.0
            elif ac.has_tail:
                margin_frac = 1.0
            else:
                margin_frac = max(-1.0, min(1.0, ac.margin / max(1, ac.total_steps)))
            ctx[consts.ACTION_ORDER[coord]] = margin_frac
        return ctx

    def _create_close_food_ctx(self, env_data: EnvMetaData, step_data: EnvStepData, snake_ctx: SnakeContext) -> np.ndarray:
        ctx = self._clean_dir_ctx()
        for rot in (True, False):
            dir_tuple = get_dir_to_tile(
                step_data.map,
                env_data.width,
                env_data.height,
                snake_ctx.head,
                env_data.food_value,
                [env_data.free_value, env_data.food_value],
                clockwise=rot
            )
            action_idx = consts.ACTION_ORDER.get(dir_tuple)
            ctx[action_idx] = 1.0
        return ctx
