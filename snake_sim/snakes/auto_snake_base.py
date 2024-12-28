import math
import itertools
from collections import deque
from array import array
from statistics import mean
import numpy as np
from time import time

from snake_sim.utils import coord_op, exec_time

from snake_sim.snakes.snake import Snake

def copy_map(s_map):
    return [array('B', row) for row in s_map]


class AutoSnakeBase(Snake):

    def __init__(self, id: str, start_length: int):
        super().__init__(id, start_length)
        self.x = None
        self.y = None
        self.route: deque = deque()
        self.start_time = 0
        self.length = start_length
        self.alive_opps = []

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
        return direction


    def _update_snake_position(self, s_map, body_coords, old_tail):
        head = body_coords[0]
        if old_tail is not None:
            s_map[old_tail[1], old_tail[0]] = self.env_data.free_value
        for i in range(2):
            x, y = body_coords[i]
            s_map[y, x] = self.head_value if body_coords[i] == head else self.body_value
        return s_map


    def _get_route(self, s_map, start, end=None, target_tiles=None):
        """Returns a route from start to end or to the first found tile of target_tiles"""
        if target_tiles is None and end is None:
            raise ValueError("end and target_tiles can't both be None")
        checked = np.full((self.init_env_data.height, self.init_env_data.width), fill_value=False, dtype=bool)
        current_coords = [start]
        coord_map = {}
        coord_maps = []
        route = []
        done = False
        while current_coords:
            next_coords = []
            for coord in current_coords:
                valid_tiles = self._valid_tiles(s_map, coord)
                if end is None:
                    if coord in target_tiles:
                        route.append(coord)
                        done = True
                else:
                    if coord == end:
                        route.append(coord)
                        done = True
                for valid_coord in valid_tiles:
                    t_x, t_y = valid_coord
                    if not checked[t_y, t_x]:
                        next_coords.append(valid_coord)
                        coord_map[valid_coord] = coord
                        checked[t_y, t_x] = True
            if done:
                counter = 0
                while route[-1] != start:
                    counter += 1
                    route.append(coord_maps[-counter][route[-1]])
                return route
            elif next_coords:
                current_coords = next_coords
                coord_maps.append(coord_map)
            else:
                return None


    def print_map(self, s_map):
        for row in s_map:
            print_row = []
            for c in row:
                if c == self.env_data.free_value:
                    print_row.append(' . ')
                elif c == self.env_data.food_value:
                    print_row.append(' F ')
                elif c == self.env_data.blocked_value:
                    print_row.append(' # ')
                elif c == self.head_value:
                    print_row.append(f' A ')
                elif c == self.body_value:
                    print_row.append(' a ')
                elif c % 2 == 0:
                    print_row.append(f' X ')
                else:
                    print_row.append(f' x ')
            print(''.join(print_row))


