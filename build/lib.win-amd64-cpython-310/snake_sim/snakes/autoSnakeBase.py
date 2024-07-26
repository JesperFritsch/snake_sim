import math
import itertools
from collections import deque
from array import array
import numpy as np
from time import time

from ..utils import coord_op, exec_time

from .snake import Snake
from statistics import mean
from ..snake_env import (
        DIR_MAPPING,
        SnakeEnv
    )

def copy_map(s_map):
    return [array('B', row) for row in s_map]


class AutoSnakeBase(Snake):

    def __init__(self, id: str, start_length: int):
        super().__init__(id, start_length)
        self.env: SnakeEnv
        self.x = None
        self.y = None
        self.route: deque = deque()
        self.start_time = 0
        self.map_to_print = None
        self.length = start_length
        self.alive_opps = []

    def in_sight(self, head_coord, coord, sight_len=2):
        h_x, h_y = head_coord
        x, y = coord
        return (h_x - sight_len) <= x <= (h_x + sight_len) and (h_y - sight_len) <= y <= (h_y + sight_len)

    def pick_direction(self):
        raise NotImplementedError

    # @exec_time
    def update(self):
        # print(f'update for {self.id} step: {self.env.time_step}')
        # print(f'self coord: {self.coord}')
        self.start_time = time()
        self.update_map(self.env.map)
        self.map_to_print = self.map.copy()
        self.update_survivors()
        next_tile = self.pick_direction()
        if next_tile is None:
            next_tile = self.coord
        return coord_op(next_tile, self.coord, '-')

    def find_body_coords(self, s_map, head_coord):
        body_coords = deque()
        next_coord = head_coord
        last_next = None
        checked = [False] * (self.env.height * self.env.width)
        while True:
            last_next = next_coord
            for coord in self.neighbours(next_coord):
                x, y = coord
                if not self.env.is_inside(coord):
                    continue
                if checked[y * self.env.width + x]:
                    continue
                if s_map[y][x] == self.body_value:
                    body_coords.append(coord)
                    checked[y * self.env.width + x] = True
                    next_coord = coord
                    break
            if next_coord == last_next:
                break
        return body_coords
    def find_attack_moves(self, s_map):
        last_pos = self.body_coords[1]
        if last_pos == self.coord: return tuple()
        head_dir = DIR_MAPPING[coord_op(self.coord, last_pos, '-')]
        moves = []
        if head_dir == 'up':
            if self.x < self.env.width - 1 and s_map[self.y + 1][self.x + 1] in self.alive_opps:
                moves.append((1, 0))
            elif self.x > 0 and s_map[self.y + 1][self.x - 1] in self.alive_opps:
                moves.append((-1, 0))
        elif head_dir == 'right':
            if self.y > 0 and s_map[self.y - 1][self.x - 1] in self.alive_opps:
                moves.append((0, -1))
            elif self.y < self.env.height - 1 and s_map[self.y + 1][self.x - 1] in self.alive_opps:
                moves.append((0, 1))
        elif head_dir == 'down':
            if self.x < self.env.width - 1 and s_map[self.y - 1][self.x + 1] in self.alive_opps:
                moves.append((1, 0))
            elif self.x > 0 and s_map[self.y - 1][self.x - 1] in self.alive_opps:
                moves.append((-1, 0))
        else:
            if self.y > 0 and s_map[self.y - 1][self.x + 1] in self.alive_opps:
                moves.append((0, -1))
            elif self.y < self.env.height - 1 and s_map[self.y + 1][self.x + 1] in self.alive_opps:
                moves.append((0, 1))
        return tuple(moves)

    def show_search(self, s_map, coord=None, current=None, checked=None):
        for y in range(self.env.height):
            for x in range(self.env.width):
                if checked is not None:
                    if checked[y, x] == True:
                        s_map[y, x] = ord('#')
                if current is not None:
                    if (x, y) in current:
                        s_map[y, x] = ord('c')
                if coord is not None:
                    if (x, y) == coord:
                        s_map[y, x] = ord('A')
        return s_map

    def show_route(self, s_map, s_route):
        s_map = s_map.copy()
        if s_route is None: return
        for x, y in list(s_route)[1:]:
            s_map[y, x] = ord('x')
        return s_map

    def occupy_route(self, s_map, s_route):
        for x, y in list(s_route):
            s_map[y, x] = self.env.BLOCKED_TILE
        return s_map

    def update_snake_position(self, s_map, body_coords, old_tail):
        head = body_coords[0]
        if old_tail is not None:
            s_map[old_tail[1], old_tail[0]] = self.env.FREE_TILE
        for i in range(2):
            x, y = body_coords[i]
            s_map[y, x] = self.head_value if body_coords[i] == head else self.body_value
        return s_map

    def get_flat_map_state(self, s_map):
        FREE_TILE = self.env.FREE_TILE
        FOOD_TILE = self.env.FOOD_TILE

        mask = (s_map == FREE_TILE) | (s_map == FOOD_TILE)
        return tuple(mask.flat)

    def update_survivors(self):
        self.alive_opps = [s.head_value for s in self.env.alive_snakes]

    def get_head_coord(self, s_map, head):
        for y, row in enumerate(s_map):
            if head in row:
                return (row.index(head), y)
        return None

    def get_route(self, s_map, start, end=None, target_tiles=None):
        """Returns a route from start to end or to a target tile if target_tiles is not None"""
        if target_tiles is None and end is None:
            raise ValueError("end and target_tiles can't both be None")
        checked = np.full((self.env.height, self.env.width), fill_value=False, dtype=bool)
        current_coords = [start]
        coord_map = {}
        coord_maps = []
        route = []
        done = False
        while current_coords:
            next_coords = []
            for coord in current_coords:
                valid_tiles = self.valid_tiles(s_map, coord)
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


    def closest_apple_route(self, current_coords, s_map, checked=None, depth=0, head_coord=None):
        if checked is None:
            checked = [False] * (self.env.height * self.env.width)
        if depth == 0:
            head_coord = current_coords[0]
        next_coords = []
        coord_map = {}
        for coord in current_coords:
            x, y = coord
            valid_tiles = self.valid_tiles(s_map, coord)
            if s_map[y, x] == self.env.FOOD_TILE:
                if len(self.valid_tiles(s_map, coord, head_coord)) >= 2:
                    return [coord]
            for valid_coord in valid_tiles:
                t_x, t_y = valid_coord
                if not checked[t_y * self.env.width + t_x]:
                    next_coords.append(valid_coord)
                    coord_map[valid_coord] = coord
                    checked[t_y * self.env.width + t_x] = True
        if next_coords:
            sub_route = self.closest_apple_route(next_coords, s_map, checked=checked, depth=depth+1, head_coord=head_coord)
            if sub_route is not None:
                if depth > 0:
                    return sub_route + [coord_map[sub_route[-1]]]
                else:
                    return sub_route
        else:
            return None

    def get_distance(self, coord1, coord2):
        x1, y1 = coord1
        x2, y2 = coord2
        x_res, y_res = coord_op(coord1, coord2, '-')
        return math.sqrt(math.pow(x_res, 2) + math.pow(y_res, 2))

    def print_map(self, s_map):
        for row in s_map:
            print_row = []
            for c in row:
                if c == self.env.FREE_TILE:
                    print_row.append(' . ')
                elif c == self.env.FOOD_TILE:
                    print_row.append(' F ')
                elif c == self.env.BLOCKED_TILE:
                    print_row.append(' # ')
                else:
                    print_row.append(f' {chr(c)} ')
            print(''.join(print_row))



    def get_areas(self, s_map, s_coord):
        dirs = [d for d in DIR_MAPPING]
        corners = [coord_op(dirs[i-1], dirs[i%len(dirs)], '+') for i in range(len(dirs))]
        corners = [c for c in set(corners) if c != (0, 0)]
        areas = []
        subareas = set()
        for corner in corners:
            x, y = coord_op(s_coord, corner, '+')
            c_x, c_y = corner
            tile_a, tile_b = coord_op(s_coord, (c_x, 0), '+'), coord_op(s_coord, (0, c_y), '+')
            corner_conn = (tile_a, tile_b)
            if all([self.env.is_inside(t) for t in corner_conn]) \
                and all([s_map[t[1], t[0]] in self.env.valid_tile_values for t in corner_conn]) \
                and s_map[y, x] in self.env.valid_tile_values:
                subareas.add(corner_conn)
            else:
                for x, y in corner_conn:
                    if self.env.is_inside((x, y)) and s_map[y, x] in self.env.valid_tile_values:
                        subareas.add(((x, y),))
        change = True
        areas_set = set()
        while change:
            change = False
            last_areas = areas_set.copy()
            areas = list(areas_set)
            for subarea in subareas:
                change = False
                handled = False
                for tile in subarea:
                    for j, area in enumerate(areas):
                        if tile in area:
                            handled = True
                            areas[j] = tuple(set(list(subarea) + list(area)))
                if not handled:
                    if subarea not in areas:
                        areas.append(subarea)
            areas_set = set(areas)
            if areas_set != last_areas:
                change = True
        return areas


    def get_areas_fast(self, s_map, s_coord, valid_tiles):
        if (max(valid_tiles, key=lambda x: x[0])[0] - min(valid_tiles, key=lambda x: x[0])[0]) == 2:
            tiles = sorted(valid_tiles, key=lambda x: x[0])
        else:
            tiles = sorted(valid_tiles, key=lambda x: x[1])
        areas = {0: [tiles[0]]}
        a = 0
        for i in range(1, len(tiles)):
            coord1, coord2 = tiles[i-1:i+1]
            dir_1 = coord_op(coord1, s_coord, '-')
            dir_2 = coord_op(coord2, s_coord, '-')
            cor_dir = coord_op(dir_1, dir_2, '+')
            cor_coord = coord_op(s_coord, cor_dir, '+')
            c_x, c_y = cor_coord
            if s_map[c_y, c_x] in self.env.valid_tile_values:
                areas[a] = areas[a] + [coord2]
            else:
                a += 1
                areas[a] = areas.get(a, []) + [coord2]
        return areas

    #
    def valid_tiles(self, s_map, coord, discount=None):
        dirs = []
        for direction in DIR_MAPPING:
            m_coord = coord_op(coord, direction, '+')
            x_move, y_move = m_coord
            if m_coord == discount:
                dirs.append(m_coord)
            elif not self.env.is_inside(m_coord):
                continue
            elif s_map[y_move, x_move] not in self.env.valid_tile_values:
                continue
            dirs.append(m_coord)
        return dirs

    def neighbours(self, coord):
        x, y = coord
        return [(x, y-1),(x+1, y),(x, y+1),(x-1, y)]


    def is_single_area(self, coord, areas):
        for area in areas:
            if coord in area and len(area) == 1:
                return True
        return False

    def calc_immediate_risk(self, s_map, option_coord):
        def coords_unique(tuples):
            coords = [coord for _, coord in tuples]
            return len(set(coords)) == len(coords)

        def get_next_states(s_map, self_coord):
            op_valid_tiles = {}
            for snake in [s for s in self.env.snakes.values() if s.alive and s is not self]:
                h_coord = snake.coord
                if self.in_sight(self_coord, h_coord, 2):
                    op_valids = [c for c in self.valid_tiles(s_map, h_coord) if self.in_sight(self_coord, c, 2)]
                    op_valid_tiles[h_coord] = {}
                    op_valid_tiles[h_coord]['tiles'] = op_valids
                    op_valid_tiles[h_coord]['head_value'] = snake.head_value
                    op_valid_tiles[h_coord]['body_value'] = snake.body_value
            options_tuples = [[(obj['head_value'], c) for c in obj['tiles']] for op_c, obj in op_valid_tiles.items()]
            options_tuples = [t for t in options_tuples if len(t) != 0]
            next_states = []
            if options_tuples:
                for op_head, data_obj in op_valid_tiles.items():
                    curr_x, curr_y = op_head
                    s_map[curr_y, curr_x] = data_obj['body_value']
                if len(op_valid_tiles) > 1:
                    combinations = [c for c in itertools.product(*options_tuples)]
                    combinations = [c for c in combinations if coords_unique(c)]
                else:
                    combinations = [(c,) for c in options_tuples[0]]
                for comb in combinations:
                    s_map_copy = copy_map(s_map)
                    for op_option in comb:
                        head_val, coord = op_option
                        op_x, op_y = coord
                        s_map_copy[op_y, op_x] = head_val
                    next_states.append(s_map_copy)
            return next_states

        def recurse(s_map, self_coord, body_coords, self_length, depth=0):
            if depth >= self.MAX_RISK_CALC_DEPTH:
                return 0
            results = []
            if s_map[self_coord[1], self_coord[0]] == self.env.FOOD_TILE:
                self_length += 1
            old_tail = self.update_body(self_coord, body_coords, self_length)
            for next_state_map in get_next_states(s_map.copy(), self_coord):
                if valids_in_next := self.valid_tiles(next_state_map, self_coord):
                    sub_results = []
                    for self_valid in valids_in_next:
                        next_state_map = self.update_snake_position(next_state_map.copy(), body_coords, old_tail)
                        result = recurse(next_state_map, self_valid, body_coords.copy(), self_length, depth+1)
                        sub_results.append(result)
                    results.append(mean(sub_results))
                else:
                    results.append(1)
            if results:
                return mean(results)
            else:
                return 0

        self_length = self.length
        if s_map[option_coord[1], option_coord[0]] == self.env.FOOD_TILE:
                self_length += 1
        body_coords = self.body_coords.copy()
        old_tail = self.update_body(option_coord, body_coords, self_length)
        s_map = self.update_snake_position(s_map, body_coords, old_tail)
        results = []
        for next_state_map in get_next_states(s_map.copy(), option_coord):
            if valids_in_next := self.valid_tiles(next_state_map, option_coord):
                sub_results = []
                for self_valid in valids_in_next:
                    result = recurse(next_state_map, self_valid, body_coords, self.length)
                    sub_results.append(result)
                results.append(mean(sub_results))
            else:
                results.append(1)
        if results:
            return mean(results)
        else:
            return 0
