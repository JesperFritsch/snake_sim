import numpy as np
from typing import Deque
from collections import deque
from snake_sim.utils import coord_op
from snake_sim.environment.types import Coord, EnvInitData, EnvData
from snake_sim.environment.interfaces.snake_interface import ISnake
from snake_sim.utils import distance


class NextStepNotImplemented(Exception):
    pass


class Snake(ISnake):
    """Almost bare minimun implementation, it will instantly die because it does not return anything from update
    But it is a useful superclass for other snake implementations because it handles the map and data provided for each step"""
    def __init__(self):
        self.id = None
        self.start_length = None
        self.alive = True
        self.body_value = None
        self.head_value = None
        self.body_coords: Deque[Coord] = deque()
        self.coord = None
        self.map = None
        self.length = self.start_length
        self.env_init_data: EnvInitData = None
        self.env_data: EnvData = None

    # from abstract class
    def set_id(self, id: int):
        self.id = id

    # from abstract class
    def set_start_length(self, start_length: int):
        self.start_length = start_length
        self.length = start_length

    # from abstract class
    def set_start_position(self, coord: Coord):
        self.x, self.y = coord
        self.coord = coord
        self.body_coords = deque([coord] * self.length)

    # from abstract class
    def set_init_data(self, env_init_data: EnvInitData):
        self.env_init_data = env_init_data
        self.head_value = self.env_init_data.snake_values[self.id]['head_value']
        self.body_value = self.env_init_data.snake_values[self.id]['body_value']

    # from abstract class
    def update(self, env_data: EnvData):
        self.env_data = env_data
        self._update_map(self.env_data.map)
        next_step = self._next_step()
        if next_step is None:
            return
        next_direction = next_step - self.coord
        self._set_new_head(next_step)
        return next_direction

    # needs to be implemented by subclass
    def _next_step(self) -> Coord:
        """ This method should return the coordinate the snake wants to move in as a Coord(x,y) tuple"""
        raise NextStepNotImplemented("The snake did not implement the _next_step method")

    def _is_inside(self, coord):
        x, y = coord
        return (0 <= x < self.env_init_data.width and 0 <= y < self.env_init_data.height)

    def _set_new_head(self, coord: Coord):
        """ Just return if the coord is invalid, the snake will be killed if it tries to move outside the map """
        if not self._is_inside(coord):
            return
        if distance(self.coord, coord) != 1:
            return
        self.x, self.y = coord
        self.coord = coord
        if self.map[self.y, self.x] == self.env_init_data.food_value:
            self.length += 1
        self._update_body(self.coord, self.body_coords, self.length)

    def _update_body(self, new_head, body_coords: deque, length):
        body_coords.appendleft(new_head)
        old_tail = None
        for _ in range(len(body_coords) - length):
            old_tail = body_coords.pop()
        return old_tail

    def _update_map(self, map: bytes):
        self.map = np.frombuffer(map, dtype=np.uint8).reshape(self.env_init_data.height, self.env_init_data.width)

    def __repr__(self) -> str:
        return f"(Class: {type(self)}, ID: {self.id}, Alive: {self.alive}, Coord: {self.coord}, Len: {self.length})"
