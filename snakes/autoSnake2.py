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
        self.start_time = time()
        self.update_map(self.env.map)
        self.map_to_print = copy_map(self.map)
        self.update_survivors()
        tile = self.pick_direction()
        print(self.body_coords)
        self.print_map(self.map_to_print)
        if tile is not None:
            self.coord = tile
            self.x, self.y = tile
            self.update_body(self.coord, self.body_coords, self.length)
            return tile
        else:
            self.alive = False
            return self.coord

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

    def get_area_info(self, s_map, body_coords, start_coord, checked=None):
        current_coords = [start_coord]
        safety_buffer = 10
        stats = {
            'area_start': start_coord,
            'food': 0,
            'tiles': 1,
            'might_escape': False,
            'needed_steps': 0,
            'total_steps': 0,
        }
        if checked is None:
            checked = np.array([False] * (self.env.height * self.env.width), dtype=bool)
        self_indexes = [0]
        body_len = len(body_coords)
        head_coord = body_coords[0]
        while current_coords:
            next_coords = []
            for coord in current_coords:
                x, y = coord
                neighbour_coords = self.neighbours(coord)
                if s_map[y][x] == self.env.FOOD_TILE:
                    stats['food'] += 1
                for coord in neighbour_coords:
                    t_x, t_y = coord
                    if self.env.is_inside(coord):
                        if not checked[t_y * self.env.width + t_x]:
                            checked[t_y * self.env.width + t_x] = True
                            if s_map[t_y][t_x] in self.env.valid_tile_values:
                                if not self.is_oneway_or_deadend(s_map, coord):
                                    stats['tiles'] += 1
                                    next_coords.append(coord)
                            elif s_map[t_y][t_x] == self.body_value:
                                self_indexes.append(body_coords.index(coord))
                current_coords = next_coords
            total_steps = stats['tiles'] - stats['food']
            max_index = max(self_indexes)
            needed_steps = body_len - max_index + safety_buffer
            stats['needed_steps'] = needed_steps
            if total_steps >= needed_steps:
                stats['might_escape'] = True
                break
        if len(self_indexes) == 1:
            #If the snake is not in the area, it might escape if the area is larger than the snake
            #Why 10? I don't know, it just works
            stats['might_escape'] = total_steps >= body_len + 10
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

    def get_option_data(self, s_map, option_coord, depth=0):
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
                'food': 0,
                'tiles': 0,
                'might_escape': False
            }
        for tile in valid_tiles:
            s_time = time()
            check_result = self.get_area_info(self.map, self.body_coords, tile)
            area_checks[tile] = check_result
        escape_options = [a_c for a_c in area_checks.values() if a_c['might_escape']]
        if escape_options:
            option['food'] = max([a_c['food'] for a_c in escape_options])
            option['might_escape'] = True
        else:
            option['food'] = max([a_c['food'] for a_c in area_checks.values()])
            option['might_escape'] = False

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

        target_option = options.get(target_tile, None)
        escape_options = [o for o in options.values() if o['might_escape']]
        best_food = max([o['food'] for o in options.values()])
        print("Target option: ", target_option)
        if target_option is not None and target_option['might_escape']:
            best_option = target_option
        elif escape_options:
            if best_food > 0:
                best_option = max(escape_options, key=lambda x: x['food'])
            else:
                best_option = max(escape_options, key=lambda x: x['tiles'])
        else:
            if best_food > 0:
                best_option = max(options.values(), key=lambda x: x['food'])
            else:
                best_option = max(options.values(), key=lambda x: x['tiles'])
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

    def get_areas(self, s_map, s_coord, valid_tiles):
        dirs = [coord_op(tile, s_coord, '-') for tile in valid_tiles]
        corners = [coord_op(dirs[i-1], dirs[i%len(dirs)], '+') for i in range(len(dirs))]
        corners = [c for c in set(corners) if c != (0, 0)]
        areas = []
        subareas = set()
        for corner in corners:
            x, y = coord_op(s_coord, corner, '+')
            c_x, c_y = corner
            tile_a, tile_b = coord_op(s_coord, (c_x, 0), '+'), coord_op(s_coord, (0, c_y), '+')
            corner_conn = (tile_a, tile_b)
            if s_map[y][x] in self.env.valid_tile_values:
                subareas.add(corner_conn)
            else:
                subareas.add((tile_a,))
                subareas.add((tile_b,))
        change = True
        while change:
            change = False
            last_areas = areas.copy()
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
            areas = list(set(areas))
            if areas != last_areas:
                change = True
        print(corners)
        print(dirs, valid_tiles)
        print('subareas: ', subareas)
        print('areas:', areas)
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

    def is_oneway_or_deadend(self, s_map, coord):
        valids = self.valid_tiles(s_map, coord)
        valids_len = len(valids)
        if valids_len == 1:
            return True
        elif valids_len == 2 and any([x == 0 for x in coord_op(valids[0], valids[1], '-')]):
            return True
        return False

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


    def recurse_check_option(self, s_map, new_coord, body_coords, length, start_time, route=None, depth=1, best_results=None, current_results=None, area_checked=False):
        if current_results is None:
            current_results = {}
            current_results['apple_time'] = []
            current_results['len_gain'] = 0
        if best_results is None:
            best_results = {}
        if s_map[new_coord[1]][new_coord[0]] == self.env.FOOD_TILE:
            length += 1
            current_results['apple_time'] = current_results['apple_time'] + [depth]
            current_results['len_gain'] = length - self.length
        current_results['depth'] = depth
        old_tail = self.update_body(new_coord, body_coords, length)
        s_map = self.update_snake_position(s_map, body_coords, old_tail)
        # if not route:
        route = self.closest_apple_route([new_coord], s_map)
        target_tile = self.target_tile(s_map, body_coords, route, recurse_mode=True)
        valid_tiles = self.valid_tiles(s_map, new_coord)
        best_results['depth'] = max(best_results.get('depth', 0), current_results['depth'])
        best_results['len_gain'] = max(best_results.get('len_gain', 0), current_results['len_gain'])
        best_results['apple_time'] = min(best_results, current_results, key=lambda x: sum(x.get('apple_time', [length*length])[:5]))['apple_time']
        valid_tiles.sort(key=lambda x: 0 if x == target_tile else 1)
        if ((time() - start_time) * 1000 > self.MAX_BRANCH_TIME) and self.TIME_LIMIT:
            best_results['timeout'] = True
            return best_results
        if current_results.get('depth', 0) >= length:
            return current_results
        # print('______________________')
        # print('new_coord: ', new_coord)
        # print('target_tile:', target_tile)
        # print('valid_tiles: ', valid_tiles)
        # print('recurse_map')
        # print(route)
        # self.print_map(s_map)
        # print('recurse_time: ', (time() - self.start_time) * 1000)
        if valid_tiles:
            s_time = time()
            areas = self.get_areas(s_map, new_coord, valid_tiles)
            # print('areas time: ', (time() - s_time) * 1000)
            # print('areas:', areas)
            area_checks = {}
            if (len(areas) > 1 or len(areas[0]) == 1):
                if not area_checked:
                    area_checked = True
                    for tiles in areas.values():
                        tile = tiles[0]
                        s_time = time()
                        check_result = self.get_area_info(s_map, body_coords, tile)
                        for t in tiles:
                            area_checks[t] = check_result
                        # print('areas_info: ', tile, area_checks[tile])
                        # print('area_info time: ', (time() - s_time) * 1000)
            else:
                area_checked = False
            for tile in valid_tiles:
                if area_check := area_checks.get(tile, {}):
                    if not area_check['might_escape']:
                        # print(f'{tile}: This area should not be entered')
                        continue
                    else:
                        return {
                            'depth': depth + area_check['tiles'],
                            'len_gain': current_results['len_gain'] + area_check['food'],
                            'apple_time': current_results['apple_time']
                        }
                check_result = self.recurse_check_option(
                    copy_map(s_map),
                    tile,
                    body_coords.copy(),
                    length,
                    route=route if tile == target_tile else None,
                    depth=depth+1,
                    best_results=best_results,
                    current_results=current_results.copy(),
                    start_time=start_time,
                    area_checked=area_checked)
                if check_result.get('depth', 0) >= length or check_result.get('timeout', False):
                    return check_result
        return best_results


