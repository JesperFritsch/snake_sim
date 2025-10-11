from abc import abstractmethod, ABC
import numpy as np
from typing import Deque
from collections import deque
from snake_sim.environment.types import Coord, EnvMetaData, EnvStepData
from snake_sim.environment.interfaces.snake_interface import ISnake
from snake_sim.utils import distance
from snake_sim.map_utils.general import print_map


class NextStepNotImplemented(Exception):
    pass


class SnakeBase(ISnake, ABC):
    """Almost bare minimum implementation, it will instantly die because it does not return anything from update
    But it is a useful superclass for other snake implementations because it handles the map and data provided for each step"""
    def __init__(self):
        super().__init__()
        self._id = None
        self._start_length = None
        self._alive = True
        self._body_value = None
        self._head_value = None
        self._body_coords: Deque[Coord] = deque()
        self._head_coord = None
        self._map = None
        self._length = self._start_length
        self._env_meta_data: EnvMetaData = None
        self._env_step_data: EnvStepData = None

    # from abstract class
    def update(self, env_step_data: EnvStepData):
        self._env_step_data = env_step_data
        self._update_map(self._env_step_data.map)
        next_step = self._next_step()
        if next_step is None:
            return
        next_direction = next_step - self._head_coord
        self._set_new_head(next_step)
        return next_direction

    @abstractmethod
    def _next_step(self) -> Coord:
        pass

    def _is_inside(self, coord):
        x, y = coord
        return (0 <= x < self._env_meta_data.width and 0 <= y < self._env_meta_data.height)

    def _set_new_head(self, coord: Coord):
        """ Just return if the coord is invalid, the snake will be killed if it tries to move outside the map """
        if not self._is_inside(coord):
            raise ValueError(f"Snake: {self._id} head at: {self._head_coord} tried to set its new head outside the map at: {coord}")
        if distance(self._head_coord, coord) != 1:
            raise ValueError(f"Snake: {self._id} head at: {self._head_coord} tried to set its new head not ajacent to head at: {coord}")
        self.x, self.y = coord
        self._head_coord = coord
        grow = False
        if self._map[self.y, self.x] == self._env_meta_data.food_value:
            self._length += 1
            grow = True
        self._move_forward(self._head_coord, self._body_coords, grow)

    def _move_forward(self, new_head: Coord, body_coords: Deque[Coord], grow: bool):
        body_coords.appendleft(new_head)
        if not grow:
            body_coords.pop()

    def _move_backwards(self, old_tail: Coord, body_coords: Deque[Coord], shrink: bool):
        body_coords.popleft()
        if not shrink:
            body_coords.append(old_tail)

    def _update_map(self, map: bytes):
        self._map = np.frombuffer(map, dtype=self._env_meta_data.base_map_dtype).reshape(self._env_meta_data.height, self._env_meta_data.width)

    def _print_map(self, s_map=None):
        print_map(
            s_map if s_map is not None else self._map,
            self._env_meta_data.free_value,
            self._env_meta_data.food_value,
            self._env_meta_data.blocked_value,
            self._head_value,
            self._body_value
        )

    def __repr__(self) -> str:
        return f"(Class: {type(self)}, ID: {self._id}, Alive: {self._alive}, Coord: {self._head_coord}, Len: {self._length})"
