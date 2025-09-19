
from abc import ABC, abstractmethod
from snake_sim.environment.types import Coord, EnvInitData, EnvData
from typing import Tuple, Deque
from collections import deque


class ISnake(ABC):

    def __init__(self):
        self._id = None
        self._start_length = None
        self._alive = True
        self._body_value = None
        self._head_value = None
        self._body_coords: Deque[Coord] = deque()
        self._head_coord = None
        self._map = None
        self._length = self._start_length
        self._env_init_data: EnvInitData = None
        self._env_data: EnvData = None

    def kill(self):
        self._alive = False

    def set_id(self, id: int):
        self._id = id

    def set_start_length(self, start_length: int):
        self._start_length = start_length
        self._length = start_length

    def set_start_position(self, coord: Coord):
        self.x, self.y = coord
        self._head_coord = coord
        self._body_coords = deque([coord] * self._length)

    def set_init_data(self, env_init_data: EnvInitData):
        self._env_init_data = env_init_data
        self._head_value = self._env_init_data.snake_values[self._id]['head_value']
        self._body_value = self._env_init_data.snake_values[self._id]['body_value']

    @abstractmethod
    def update(self, env_data: EnvData) -> Coord: # -> (int, int) as direction (1, 0) for right (-1, 0) for left
        pass

