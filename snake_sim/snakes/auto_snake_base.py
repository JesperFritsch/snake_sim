import math
import itertools
from collections import deque
from array import array
from statistics import mean
import numpy as np
from time import time

from snake_sim.utils import coord_op, exec_time
from snake_sim.environment.types import Coord

from snake_sim.snakes.snake import Snake

def copy_map(s_map):
    return [array('B', row) for row in s_map]


class AutoSnakeBase(Snake):

    def __init__(self):
        super().__init__()
        self.x = None
        self.y = None
        self.route: deque = deque()
        self.start_time = 0
        self.length = None
        self.alive_opps = []

    def set_start_length(self, start_length):
        super().set_start_length(start_length)
        self.length = start_length

    def _pick_direction(self):
        raise NotImplementedError

    def update(self, env_data: dict):
        super().update(env_data)
        self.start_time = time()
        next_tile = self._pick_direction()
        if next_tile is None:
            next_tile = self.coord
        direction = coord_op(next_tile, self.coord, '-')
        self.set_new_head(next_tile)
        return Coord(*direction)


    def _update_snake_position(self, s_map, body_coords, old_tail):
        head = body_coords[0]
        if old_tail is not None:
            s_map[old_tail[1], old_tail[0]] = self.env_data.free_value
        for i in range(2):
            x, y = body_coords[i]
            s_map[y, x] = self.head_value if body_coords[i] == head else self.body_value
        return s_map





