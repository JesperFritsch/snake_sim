import random
import math
import itertools
import numpy as np
# from numba import jit
from collections import deque
from array import array
from time import time
from dataclasses import dataclass, field

from snakes.snake import Snake
from statistics import mean
from snake_env import (
        coord_op,
        DIR_MAPPING
    )

def copy_map(s_map):
    return [array('B', row) for row in s_map]


class AutoSnake2(Snake):
    TIME_LIMIT = True
    MAX_RISK_CALC_DEPTH = 3
    MAX_BRANCH_TIME = 1000


    def __init__(self, id: str, start_length: int):
        super().__init__(id, start_length)
        self.env
        self.x = None
        self.y = None
        self.route = None
        self.start_time = 0
        self.map_to_print = None
        self.length = start_length
        self.alive_opps = []

    def in_sight(self, head_coord, coord, sight_len=2):
        h_x, h_y = head_coord
        x, y = coord
        return (h_x - sight_len) <= x <= (h_x + sight_len) and (h_y - sight_len) <= y <= (h_y + sight_len)

    def update(self):
        print(f'update for {self.id} step: {self.env.time_step}')
        print(f'self coord: {self.coord}')
        self.start_time = time()
        self.update_map(self.env.map)
        self.map_to_print = copy_map(self.map)
        print(self.body_coords)
        self.print_map(self.map_to_print)
        self.update_survivors()
        tile = self.pick_direction()
        if tile is not None:
            self.coord = tile
            self.x, self.y = tile
            self.update_body(self.coord, self.body_coords, self.length)
            next_tile = tile
        else:
            self.alive = False
            next_tile = self.coord
        return next_tile

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

    def show_route(self, s_map, s_route):
        if s_route is None: return
        dest = s_route[0]
        for x, y in s_route[1:]:
            s_map[y][x] = ord('¤')
        return copy_map(s_map)

    def show_search(self, s_map, checked, coord, current):
        for y in range(self.env.height):
            for x in range(self.env.width):
                if checked[y * self.env.width + x]:
                    s_map[y][x] = ord('#')
                if (x, y) in current:
                    s_map[y][x] = ord('x')
                if (x, y) == coord:
                    s_map[y][x] = ord('X')
        return s_map

    #
    def update_snake_position(self, s_map, body_coords, old_tail):
        head = body_coords[0]
        if old_tail is not None:
            s_map[old_tail[1]][old_tail[0]] = self.env.FREE_TILE
        for i in range(2):
            x, y = body_coords[i]
            s_map[y][x] = self.head_value if body_coords[i] == head else self.body_value
        return s_map

    def update_map(self, flat_map):
        if self.map is None:
            self.map = [array('B', [self.env.FREE_TILE] * self.env.width) for _ in range(self.env.height)]
        for y in range(self.env.height):
            for x in range(self.env.width):
                map_val = flat_map[y * self.env.width + x]
                self.map[y][x] = map_val

    def update_survivors(self):
        self.alive_opps = [s.head_value for s in self.env.alive_snakes]

    def get_head_coord(self, s_map, head):
        for y, row in enumerate(s_map):
            if head in row:
                return (row.index(head), y)
        return None

    def is_single_area(self, coord, areas):
        for area in areas:
            if coord in area and len(area) == 1:
                return True
        return False

    def get_area_info(self, s_map, body_coords, start_coord, checked=None):
        current_coords = [start_coord]
        safety_buffer = 10
        stats = {
            'area_start': start_coord,
            'food': 0,
            'tiles': 1,
            'might_escape': False,
            'needed_steps': 0,
            'total_steps': 1,
        }
        if checked is None:
            checked = np.array([False] * (self.env.height * self.env.width), dtype=bool)
        self_indexes = [0]
        body_len = len(body_coords)
        while current_coords:
            next_coords = []
            # print('current_coords: ', current_coords)
            for curr_coord in current_coords:
                # print('___________________________')
                c_x, c_y = curr_coord
                if s_map[c_y][c_x] == self.env.FOOD_TILE:
                    stats['food'] += 1
                areas = self.get_areas(s_map, curr_coord)
                # print('areas:', areas)
                unexplored_areas = [a for a in areas if not any([checked[t[1] * self.env.width + t[0]] for t in a])]
                # print('unexp:', unexplored_areas)
                # print('coord: ', curr_coord)
                # print('unexp:', unexplored_areas)
                # print('areas:', areas)
                # checked_map = self.show_search(copy_map(s_map), checked, curr_coord, current_coords)
                # self.print_map(checked_map)
                for n_coord in self.neighbours(curr_coord):
                    n_x, n_y = n_coord
                    # print('neighbour:', n_coord)
                    if self.is_single_area(n_coord, areas):
                        # print('n_coord:', n_coord)
                        # print(coord)
                        # print('unexp:', unexplored_areas)
                        # print('areas:', areas)
                        # checked_map = self.show_search(copy_map(s_map), checked, coord, current_coords)
                        # self.print_map(checked_map)
                        if not checked[n_y * self.env.width + n_x]:
                            # print('doing subsearch')
                            checked[n_y * self.env.width + n_x] = True
                            area_info = self.get_area_info(s_map, body_coords, n_coord, checked=checked)
                            if area_info['might_escape']:
                                stats['might_escape'] = True
                                # stats['needed_steps'] = area_info['needed_steps']
                                stats['total_steps'] += area_info['total_steps']
                                stats['food'] += area_info['food']
                                stats['tiles'] += area_info['tiles']
                            # print('stats: ', stats)
                            # print('area_info:', area_info)
                    else:
                        t_x, t_y = n_coord
                        if self.env.is_inside(n_coord):
                            # print('is_inside')
                            if not checked[t_y * self.env.width + t_x]:
                                # print('is_not_checked')
                                if s_map[t_y][t_x] in self.env.valid_tile_values:
                                    checked[t_y * self.env.width + t_x] = True
                                    # print('is_valid')
                                    stats['tiles'] += 1
                                    next_coords.append(n_coord)
                                elif s_map[t_y][t_x] == self.body_value:
                                    # print('is_body')
                                    self_indexes.append(body_coords.index(n_coord))
                checked[c_y * self.env.width + c_x] = True
            current_coords = next_coords
            total_steps = stats['tiles'] - stats['food']
            max_index = max(self_indexes)
            needed_steps = body_len - max_index + safety_buffer
            stats['needed_steps'] = needed_steps
            stats['total_steps'] = total_steps
            if total_steps >= needed_steps:
                stats['might_escape'] = True
                break

        # print('returning stats: ', stats)
        return stats



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
            if s_map[y][x] == self.env.FOOD_TILE:
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

    def get_option_data(self, s_map, option_coord):
        # t_dir = coord_op(option_coord, self.coord, '-')
        body_coords = self.body_coords.copy()
        old_tail = self.update_body(option_coord, body_coords, self.length)
        s_map = self.update_snake_position(copy_map(s_map), body_coords, old_tail)
        valid_tiles = self.valid_tiles(s_map, option_coord)
        time_s = time()
        option = {'coord': option_coord}
        area_checks = {}
        if not valid_tiles:
            return {
                'coord': option_coord,
                'food': 0,
                'tiles': 0,
                'might_escape': False,
                'needed_steps': 0,
                'risk': 0
            }
        for tile in valid_tiles:
            s_time = time()
            check_result = self.get_area_info(s_map, self.body_coords, tile)
            area_checks[tile] = check_result
        escape_options = [a_c for a_c in area_checks.values() if a_c['might_escape']]
        if escape_options:
            option['food'] = max([a_c['food'] for a_c in escape_options])
            option['might_escape'] = True
            option['needed_steps'] = min([a_c['needed_steps'] for a_c in escape_options])
        else:
            option['food'] = max([a_c['food'] for a_c in area_checks.values()])
            option['might_escape'] = False
            option['needed_steps'] = min([a_c['needed_steps'] for a_c in area_checks.values()])

        option['tiles'] = max([a_c['tiles'] for a_c in area_checks.values()])
        option['risk'] = self.calc_immediate_risk(copy_map(s_map), option_coord)
        return option

    def pick_direction(self):
        valid_tiles = self.valid_tiles(self.map, self.coord)
        random.shuffle(valid_tiles)
        options = {}
        time_s = time()
        self.route = self.closest_apple_route([self.coord], self.map)
        # print(f"Route time:", (time() - time_s) * 1000)
        target_tile = self.target_tile(self.map, self.body_coords, self.route)
        route_copy = None
        best_option = None
        if self.route is not None:
            route_copy = self.route + [target_tile]
        self.show_route(self.map_to_print, route_copy)
        for tile in valid_tiles:
            option_data = self.get_option_data(copy_map(self.map), tile)
            options[tile] = option_data
        for option in options:
            print(option, options[option])
        if options:
            target_option = options.get(target_tile, None)
            escape_options = [o for o in options.values() if o['might_escape']]
            best_food_option = max(options.values(), key=lambda x: x['food'])
            best_food = best_food_option['food']
            lowest_risk = min([o['risk'] for o in options.values()])
            lowest_risk_options = [o for o in options.values() if o['risk'] == lowest_risk]
            # print("Target option: ", target_option)
            if any([o['risk'] != 0 for o in options.values()]):
                if any([o['might_escape'] for o in lowest_risk_options]):
                    best_option = max(lowest_risk_options, key=lambda x: x['might_escape'])
                else:
                    best_option = min(lowest_risk_options, key=lambda x: x['needed_steps'])
            elif target_option is not None and target_option['might_escape']:
                best_option = target_option
            elif escape_options:
                if best_food > 0:
                    best_option = max(escape_options, key=lambda x: x['food'])
                else:
                    best_option = min(escape_options, key=lambda x: x['needed_steps'])
            else:
                if best_food > 0:
                    best_option = max(options.values(), key=lambda x: x['food'])
                else:
                    best_option = min(options.values(), key=lambda x: x['needed_steps'])
            print(f"best_option: {best_option}")
        if best_option is not None:
            return best_option['coord']
        return None


    def target_tile(self, s_map, body_coords, route, recurse_mode=False):
        self_coord = body_coords[0]
        if not recurse_mode and (attack_moves := self.find_attack_moves(s_map)):
            return coord_op(self_coord, attack_moves[0], '+')
        if route:
            target_tile = route.pop()
            if not route: route = None
            return target_tile
        s_dir = coord_op(self_coord, body_coords[1], '-')
        return coord_op(self_coord, s_dir, '+')

    def print_map(self, s_map):
        for row in s_map:
            print(''.join([f' {chr(c)} ' for c in row]))

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
                and all([s_map[t[1]][t[0]] in self.env.valid_tile_values for t in corner_conn]) \
                and s_map[y][x] in self.env.valid_tile_values:
                subareas.add(corner_conn)
            else:
                for x, y in corner_conn:
                    if self.env.is_inside((x, y)) and s_map[y][x] in self.env.valid_tile_values:
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
            elif s_map[y_move][x_move] not in self.env.valid_tile_values:
                continue
            dirs.append(m_coord)
        return dirs


    def neighbours(self, coord):
        x, y = coord
        return [(x, y-1),(x+1, y),(x, y+1),(x-1, y)]


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
                    s_map[curr_y][curr_x] = data_obj['body_value']
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
                        s_map_copy[op_y][op_x] = head_val
                    next_states.append(s_map_copy)
            return next_states

        def recurse(s_map, self_coord, body_coords, self_length, depth=0):
            if depth >= self.MAX_RISK_CALC_DEPTH:
                return 0
            results = []
            if s_map[self_coord[1]][self_coord[0]] == self.env.FOOD_TILE:
                self_length += 1
            old_tail = self.update_body(self_coord, body_coords, self_length)
            for next_state_map in get_next_states(copy_map(s_map), self_coord):
                if valids_in_next := self.valid_tiles(next_state_map, self_coord):
                    sub_results = []
                    for self_valid in valids_in_next:
                        next_state_map = self.update_snake_position(copy_map(next_state_map), body_coords, old_tail)
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
        if s_map[option_coord[1]][option_coord[0]] == self.env.FOOD_TILE:
                self_length += 1
        body_coords = self.body_coords.copy()
        old_tail = self.update_body(option_coord, body_coords, self_length)
        s_map = self.update_snake_position(s_map, body_coords, old_tail)
        results = []
        for next_state_map in get_next_states(copy_map(s_map), option_coord):
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

