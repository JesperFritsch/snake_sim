from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Deque, Dict


# from snake_sim.environment.interfaces.snake_strategy_interface import ISnakeStrategy
from snake_sim.environment.interfaces.snake_interface import ISnake
from snake_sim.environment.types import Coord, EnvInitData, EnvData


class IStrategySnake(ISnake, ABC):

    """An interface for snakes that use strategies to determine their next move."""

    def __init__(self):
        super().__init__()
        self._strategies: Dict[int, 'ISnakeStrategy'] = {}

    def set_strategy(self, prio: int, strategy: 'ISnakeStrategy'):
        strategy.set_snake(self)
        self._strategies[prio] = strategy

    def _get_strategy_tile(self) -> Coord:
        for prio in sorted(self._strategies.keys()):
            strat = self._strategies[prio]
            tile = strat.get_wanted_tile()
            if tile is not None:
                return tile
        return Coord(0,0)

    def get_env_init_data(self) -> EnvInitData:
        return self._env_init_data
    
    def get_env_data(self) -> EnvData:
        return self._env_data
    
    def get_self_map_values(self) -> Tuple[int, int]: # head_value, body_value
        return self._head_value, self._body_value

    def get_head_coord(self) -> Coord:
        return self._head_coord

    def get_body_coords(self) -> Deque[Coord]:
        return self._body_coords

    def get_map(self) -> np.ndarray:
        return self._map