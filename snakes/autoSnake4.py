import random
import numpy as np
import itertools
from collections import deque
from statistics import mean
from time import time

from snakes.autoSnakeBase import AutoSnakeBase, copy_map
from snake_env import (
        coord_op,
        DIR_MAPPING
    )



class AutoSnake4(AutoSnakeBase):
    TIME_LIMIT = True
    MAX_RISK_CALC_DEPTH = 3
    MAX_BRANCH_TIME = 3000


    def __init__(self, id: str, start_length: int):
        super().__init__(id, start_length)
        self.food_in_route = []

    def set_route(self, route: deque):
        if not route:
            self.route = None
            return
        if self.coord in route:
            route = deque(list(route)[route.index(self.coord)+1:])
        self.food_in_route = []
        for coord in route:
            if self.map[coord[1]][coord[0]] == self.env.FOOD_TILE:
                self.food_in_route.append(coord)
        self.route = route

    def refine_route(self, route: deque):
        s_map = copy_map(self.map)
        body_coords = self.body_coords.copy()
        route_list = list(route)
        new_route = None

        for food in sorted(self.food_in_route, key=lambda x: route_list.index(x)):
            f_x, f_y = food
            if s_map[f_y][f_x] != self.env.FOOD_TILE and food in route_list and food in self.food_in_route:
                food_index = self.food_in_route.index(food)
                start_search_index = 0
                if 0 < food_index < len(self.food_in_route)-1:
                    food_before = self.food_in_route[food_index - 1]
                    food_after = self.food_in_route[food_index + 1]
                    food_before_index = route_list.index(food_before)
                    for coord in route_list[:food_before_index]:
                        old_tail = self.update_body(coord, body_coords, self.length)
                        s_map = self.update_snake_position(s_map, body_coords, old_tail)
                    start_search_index = food_before_index

                elif food_index == 0 and len(self.food_in_route) > 1:
                    food_after = self.food_in_route[1]

                elif food in route_list:
                    food_before = self.food_in_route[food_index - 1]
                    food_before_index = route_list.index(food_before)
                    new_route = route_list[:food_before_index]
                    start_search_index = len(route_list)

                for coord in route_list[start_search_index:]:
                    old_tail = self.update_body(coord, body_coords, self.length)
                    s_map = self.update_snake_position(s_map, body_coords, old_tail)
                    if new_subroute := self.get_route(s_map, coord, food_after):
                        sub_begin = new_subroute[0]
                        sub_end = new_subroute[-1]
                        sub_begin_index = route_list.index(sub_begin)
                        sub_end_index = route_list.index(sub_end)
                        new_route = route_list[:sub_begin_index] + new_subroute + route_list[sub_end_index+1:]
                if food in self.food_in_route:
                    self.food_in_route.remove(food)
            if new_route:
                self.set_route(deque(new_route))


    def find_route(self, s_map, body_coords, head_coord, option_coord):
        t_dir = coord_op(option_coord, head_coord, '-')
        time_s = time()
        route = self.deep_look_ahead(copy_map(s_map), option_coord, body_coords.copy(), self.length, start_time=time_s)
        route['coord'] = option_coord
        route['timeout'] = route.get('timeout', False)
        route['free_path'] = route['depth'] >= self.length
        route['dir'] = t_dir
        time_s = time()
        route['risk'] = self.calc_immediate_risk(copy_map(s_map), option_coord)
        return route

    def get_best_route(self):
        options = {}
        best_option = None
        valid_tiles = self.valid_tiles(self.map, self.coord)
        for coord in valid_tiles:
            option = self.find_route(self.map, self.body_coords, self.coord, coord)
            options[coord] = option
        free_options = [o for o in options.values() if o['free_path']]
        # print(f'{options=}')
        if options:
            if risk_free_options := [o for o in options.values() if o['risk'] == 0]:
                if free_riskless_options := [o for o in risk_free_options if o['free_path']]:
                    options_considered = free_riskless_options
                #If there are no free path options, consider the ones that timed out
                elif timedout_options := [o for o in risk_free_options if o['timeout']]:
                    options_considered = timedout_options
                else:
                    options_considered = risk_free_options
                best_len_gain = max(o['len_gain'] for o in options_considered)
                best_len_gain = min(best_len_gain, 10)
                best_len_opts = [o for o in options_considered if o['len_gain'] >= best_len_gain]
                best_early_gain_opt = min(best_len_opts, key=lambda o: sum(o['apple_time'][:best_len_gain]))
                best_option = best_early_gain_opt
            else:
                if free_options:
                    best_option = min(free_options, key=lambda x: x['risk'])
                else:
                    best_len_gain = max(o['len_gain'] for o in options.values())
                    if best_len_gain == 0:
                        best_option = max(options.values(), key=lambda x: x['depth'])
                    else:
                        best_option = max(options.values(), key=lambda x: x['len_gain'])
        if best_option is not None:
            return best_option['body_coords']
        return None

    def pick_direction(self):
        next_tile = None
        if self.verify_route(self.route):
            # print('route verified')
            next_tile = self.route[-1]
            option = self.deep_look_ahead(copy_map(self.map), next_tile, self.body_coords.copy(), self.length, planned_route=self.route)
            # print(option)
            self.set_route(option['body_coords'])
            # print('extended route: ', self.route)
            if self.route: next_tile = self.route.pop()
        else:
            # print('route not verified')
            route = self.get_best_route()
            self.set_route(route)
            # print('new route: ', self.route)
            if self.route:
                next_tile = self.route.pop()
            else:
                next_tile = None
        return next_tile

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
            if s_map[c_y][c_x] in self.env.valid_tile_values and not coord_op(dir_1, dir_2, '+') == (0, 0):
                areas[a] = areas[a] + [coord2]
            else:
                a += 1
                areas[a] = areas.get(a, []) + [coord2]
        return list(areas.values())


    def is_single_area(self, coord, areas):
        for area in areas:
            if coord in area and len(area) == 1:
                return True
        return False

    def is_area_clear(self, s_map, body_coords, start_coord, checked=None):
        current_coords = [start_coord]
        safety_buffer = 0
        if checked is None:
            checked = np.array([False] * (self.env.height * self.env.width), dtype=bool)
        body_len = len(body_coords)
        self_indexes = [0]
        tile_count = 1
        food_count = 0
        is_clear = False
        while current_coords:
            next_coords = []
            # print('current_coords: ', current_coords)
            for curr_coord in current_coords:
                # print('___________________________')
                c_x, c_y = curr_coord
                checked[c_y * self.env.width + c_x] = True
                # print('coord: ', curr_coord)
                # checked_map = self.show_search(copy_map(s_map), checked, curr_coord, current_coords)
                # self.print_map(checked_map)
                for n_coord in self.neighbours(curr_coord):
                    # print('neighbour:', n_coord)
                    t_x, t_y = n_coord
                    if self.env.is_inside(n_coord):
                        # print('is_inside')
                        if not checked[t_y * self.env.width + t_x]:
                            if s_map[t_y][t_x] in self.env.valid_tile_values:
                                if s_map[c_y][c_x] == self.env.FOOD_TILE:
                                    food_count += 1
                                checked[t_y * self.env.width + t_x] = True
                                tile_count += 1
                                # print('is_valid')
                                next_coords.append(n_coord)
                            elif s_map[t_y][t_x] == self.body_value:
                                # print('is_body')
                                self_indexes.append(body_coords.index(n_coord))
            current_coords = next_coords
            total_steps = tile_count - food_count
            max_index = max(self_indexes)
            needed_steps = body_len - max_index + safety_buffer
            if total_steps >= needed_steps:
                is_clear = True
                break

        return is_clear

    def verify_route(self, route):
        has_route = route is not None and len(route) > 0
        if has_route:
            for coord in route:
                x, y = coord
                if self.map[y][x] not in [self.env.FREE_TILE, self.env.FOOD_TILE, self.body_value, self.head_value]:
                    return False
        return has_route


    def deep_look_ahead(self, s_map, new_coord, body_coords, length,
                        start_time=None,
                        apple_route=None,
                        planned_route=None,
                        depth=1,
                        best_results=None,
                        current_results=None,
                        area_checked=False):
        if start_time is None:
            start_time = time()
        planned_tile = None
        if current_results is None:
            current_results = {}
            current_results['timeout'] = False
            current_results['apple_time'] = []
            current_results['len_gain'] = 0
        if best_results is None:
            best_results = {}
        if s_map[new_coord[1]][new_coord[0]] == self.env.FOOD_TILE:
            length += 1
            current_results['apple_time'] = current_results['apple_time'] + [depth]
            current_results['len_gain'] = length - self.length
        current_results['body_coords'] = body_coords
        current_results['depth'] = depth
        old_tail = self.update_body(new_coord, body_coords, length)
        s_map = self.update_snake_position(s_map, body_coords, old_tail)
        valid_tiles = self.valid_tiles(s_map, new_coord)

        if planned_route:
            planned_tile = planned_route.pop()
        if planned_tile and planned_tile in valid_tiles:
            check_result = self.deep_look_ahead(
                copy_map(s_map),
                planned_tile,
                body_coords.copy(),
                length,
                depth=depth+1,
                planned_route=planned_route,
                best_results=best_results,
                current_results=current_results.copy(),
                start_time=start_time,
                area_checked=area_checked)
            if check_result.get('depth', 0) >= length or check_result['timeout']:
                return check_result

        apple_route = self.closest_apple_route([new_coord], s_map)
        target_tile = self.target_tile(s_map, body_coords, apple_route, recurse_mode=True)
        best_results['depth'] = max(best_results.get('depth', 0), current_results['depth'])
        best_results['len_gain'] = max(best_results.get('len_gain', 0), current_results['len_gain'])
        best_results['apple_time'] = []
        best_results['timeout'] = False
        best_results['body_coords'] = max(best_results.get('body_coords', deque()), current_results['body_coords'], key=lambda o: len(o))
        valid_tiles.sort(key=lambda x: 0 if x == target_tile else 1)
        if ((time() - start_time) * 1000 > self.MAX_BRANCH_TIME) and self.TIME_LIMIT:
            best_results['timeout'] = True
            return best_results
        if current_results['depth'] >= length:
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
            for tile in valid_tiles:
                area_check = self.is_area_clear(s_map, body_coords, tile)
                if not area_check:
                    continue
                check_result = self.deep_look_ahead(
                    copy_map(s_map),
                    tile,
                    body_coords.copy(),
                    length,
                    apple_route=apple_route if tile == target_tile else None,
                    depth=depth+1,
                    best_results=best_results,
                    current_results=current_results.copy(),
                    start_time=start_time,
                    area_checked=area_checked)
                if check_result['depth'] >= length or check_result.get('timeout', False):
                    return check_result
        return best_results