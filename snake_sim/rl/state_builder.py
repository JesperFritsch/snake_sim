"""State building utilities.

Goal: Provide a minimal, extensible base state representation (a dict) that downstream
algorithms (PPO, DQN, etc.) can adapt or augment without rewriting core extraction.

Channel semantics (v1):
  0 my_head
  1 my_body
  2 food
  3 other_heads
  4 other_bodies

Adapters can append channels or extend ctx; they must also increment `state['version']`
OR set `state['meta']['extended'].append(<name>)` for traceability.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol, Dict, Any, Optional

import numpy as np

from snake_sim.environment.types import EnvMetaData


BASE_STATE_VERSION = 1


@dataclass
class SnakeContext:
    snake_id: str
    head: Any  # expected to be Coord-like with x,y
    body_coords: List[Any]
    length: int


class IStateAdapter(Protocol):
    name: str
    def apply(self, state: Dict[str, Any], env_meta: EnvMetaData) -> Dict[str, Any]: ...


class BaseStateBuilder:
    def __init__(self):
        pass

    def build(self, env_meta: EnvMetaData, s_map: np.ndarray, ctx: SnakeContext) -> Dict[str, Any]:
        C = 5 
        out = np.zeros((C, env_meta.height, env_meta.width), dtype=np.float32)
        # My head / body
        out[0, ctx.head.y, ctx.head.x] = 1.0
        for bc in ctx.body_coords[1:]:  # skip head duplicate
            out[1, bc.y, bc.x] = 1.0
        # Food
        out[2][s_map == env_meta.food_value] = 1.0
        opponent_head_vals = []
        opponent_body_vals = []
        for sid, vals in env_meta.snake_values.items():
            if sid == ctx.snake_id:
                continue
            hv = vals.get('head_value')
            bv = vals.get('body_value')
            if hv is not None:
                opponent_head_vals.append(hv)
            if bv is not None:
                opponent_body_vals.append(bv)
        if opponent_head_vals:
            out[3][np.isin(s_map, opponent_head_vals)] = 1.0
        if opponent_body_vals:
            out[4][np.isin(s_map, opponent_body_vals)] = 1.0

        state = {
            'map': out,
        }
        return state


class StatePipeline:
    def __init__(self, adapters: Optional[List[IStateAdapter]] = None):
        self.adapters = adapters or []

    def run(self, state: Dict[str, Any], env_meta: Any) -> Dict[str, Any]:
        for ad in self.adapters:
            state = ad.apply(state, env_meta)
        return state


__all__ = [
    'BaseStateBuilder',
    'StatePipeline',
    'FoodDistanceAdapter',
    'SnakeContext',
    'IStateAdapter'
]
