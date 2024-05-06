import random
import numpy as np
import itertools
from collections import deque
from statistics import mean
from time import time

from utils import coord_op, coord_cmp, exec_time

from snakes.autoSnakeBase import AutoSnakeBase, copy_map
from snake_env import (
        DIR_MAPPING,
    )

class AutoSnake4(AutoSnakeBase):
    TIME_LIMIT = True
    MAX_RISK_CALC_DEPTH = 3
    MAX_BRANCH_TIME = 5000


    def __init__(self, id: str, start_length: int, greedy=False):
        super().__init__(id, start_length)
        self.greedy = greedy
        self.food_in_route = []

    def set_route(self, route: deque):
        if not route:
            self.route = None
            return
        if self.coord in route:
            route = deque(list(route)[:route.index(self.coord)])
        if not route:
            self.route = None
            return
        valid_tiles = self.valid_tiles(self.map, self.coord)
        if route[-1] not in valid_tiles:
            try:
                sub_route = self.get_route(self.map, self.coord, end=route[-1])
                route = deque(list(route) + sub_route[1:-1])
            except Exception as e:
                print(e)
                raise ValueError('Invalid route')
        self.food_in_route = []
        last_coord = None
        for coord in reversed(route):
            if coord == last_coord:
                route.remove(coord)
            if self.map[coord[1], coord[0]] == self.env.FOOD_TILE:
                self.food_in_route.append(coord)
        self.route = route

    def find_route(self, start_coord, planned_route=None, old_route=None, timeout_ms=None):
        time_s = time()
        option = self.deep_look_ahead(
                self.map.copy(),
                start_coord,
                self.body_coords.copy(),
                self.length,
                start_time=time_s,
                planned_route=planned_route,
                old_route=old_route,
                timeout_ms=timeout_ms
            )
        option['coord'] = start_coord
        # option['risk'] = self.calc_immediate_risk(self.map.copy(), start_coord)
        option['risk'] = 0
        # print(option)
        return option


    def get_best_route(self):
        options = {}
        best_option = None
        valid_tiles = self.valid_tiles(self.map, self.coord)
        for coord in valid_tiles:
            option = self.find_route(coord, planned_route=self.route)
            options[coord] = option
            # print('found option: ', option)
            if option['free_path'] and option['risk'] == 0:
                break
        free_options = [o for o in options.values() if o['free_path']]
        if options:
            if risk_free_options := [o for o in options.values() if o['risk'] == 0]:
                if free_riskless_options := [o for o in risk_free_options if o['free_path']]:
                    options_considered = free_riskless_options
                    best_len_gain = max(o['len_gain'] for o in options_considered)
                    best_len_gain = min(best_len_gain, 10)
                    best_len_opts = [o for o in options_considered if o['len_gain'] >= best_len_gain]
                    best_early_gain_opt = min(best_len_opts, key=lambda o: sum(o['apple_time'][:best_len_gain]))
                    best_option = best_early_gain_opt
                #If there are no free path options, consider the ones that timed out
                else:
                    if timedout_options := [o for o in risk_free_options if o['timeout']]:
                        options_considered = timedout_options
                    else:
                        options_considered = risk_free_options
                    best_option = max(options.values(), key=lambda x: x['depth'])
            else:
                if free_options:
                    best_option = min(free_options, key=lambda x: x['risk'])
                else:
                    # best_len_gain = max(o['len_gain'] for o in options.values())
                    best_len_gain = 0
                    if best_len_gain == 0:
                        best_option = max(options.values(), key=lambda x: x['depth'])
                    else:
                        best_option = max(options.values(), key=lambda x: x['len_gain'])
        if best_option is not None:
            return best_option['route']
        return None

    def check_safe_food_route(self, s_map, food_route):
        end_coord = food_route[0]
        self.occupy_route(s_map, food_route)
        if self.area_check(s_map, self.body_coords, end_coord)['is_clear']:
            return True
        return False

    def pick_direction(self):
        next_tile = None
        look_ahead_tile = None
        closest_food_route = self.get_route(self.map, self.coord, target_tiles=[l for l in self.env.food.locations if l != self.coord])
        if closest_food_route and ((self.check_safe_food_route(self.map.copy(), closest_food_route) and self.greedy) or \
                                    (closest_food_route[0] not in self.food_in_route)):
            planned_route = closest_food_route[:-1]
            old_route = self.route
        else:
            old_route = None
            planned_route = self.route
        if self.verify_route(planned_route):
            look_ahead_tile = planned_route.pop()
            option = self.find_route(look_ahead_tile, planned_route=planned_route, old_route=old_route, timeout_ms=300)
            if option['free_path']:
                self.set_route(option['route'])
                if self.route:
                    next_tile = self.route.pop()
                    return next_tile

        route = self.get_best_route()
        self.set_route(route)
        if self.route:
            next_tile = self.route.pop()
        else:
            next_tile = None
        return next_tile

    def target_tile(self, s_map, body_coords, recurse_mode=False):
        self_coord = body_coords[0]
        if not recurse_mode and (attack_moves := self.find_attack_moves(s_map)):
            return coord_op(self_coord, attack_moves[0], '+')
        s_dir = coord_op(self_coord, body_coords[1], '-')
        return coord_op(self_coord, s_dir, '+')

    def get_areas_fast2(self, s_map, s_coord):
        curr_area = deque()
        areas = []
        s_x, s_y = s_coord
        s_dirs = list(DIR_MAPPING)
        n_coords = ((s_x, s_y-1),(s_x+1, s_y),(s_x, s_y+1),(s_x-1, s_y))
        corner_coords = ((s_x+1, s_y-1), (s_x+1, s_y+1), (s_x-1, s_y+1), (s_x-1, s_y-1))
        # print(n_coords)
        nr_splits = 0
        i = -1
        for n_coord in n_coords:
            i += 1
            if self.env.is_inside(n_coord):
                if s_map[n_coord[1], n_coord[0]] == self.env.FREE_TILE or s_map[n_coord[1], n_coord[0]] == self.env.FOOD_TILE:
                    curr_area.append(n_coord)
                else:
                    if len(curr_area) != 0:
                        areas.append(curr_area)
                        curr_area = deque()
                        nr_splits += 1
                corner_coord = corner_coords[i]
                if self.env.is_inside(corner_coord):
                    if self.map[corner_coord[1], corner_coord[0]] != self.env.FREE_TILE and self.map[corner_coord[1], corner_coord[0]] != self.env.FOOD_TILE:
                        if len(curr_area) != 0:
                            areas.append(curr_area)
                            curr_area = deque()
                            nr_splits += 1
            else:
                if len(curr_area) != 0:
                    areas.append(curr_area)
                    curr_area = deque()
                    nr_splits += 1
        if nr_splits >= 1:
            lu_corn = (s_x-1, s_y-1)
            if self.env.is_inside(lu_corn) and (s_map[lu_corn[1], lu_corn[0]] == self.env.FREE_TILE or s_map[lu_corn[1], lu_corn[0]] == self.env.FOOD_TILE):
                areas[0].extend(areas.pop())
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
            if s_map[c_y, c_x] in self.env.valid_tile_values and not coord_op(dir_1, dir_2, '+') == (0, 0):
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

    def area_check(self, s_map, body_coords, start_coord, tile_count=1, food_count=0, checked=None, depth=0):
        current_coords = deque([start_coord])
        if checked is None:
            checked = np.full((self.env.height, self.env.width), fill_value=False, dtype=bool)
        body_len = len(body_coords)
        tail_coord = body_coords[-1]
        # self_indexes = [0]
        max_index = 0
        is_clear = False
        has_tail = False
        done = False
        total_steps = 0

        while current_coords:
            curr_coord = current_coords.popleft()
            c_x, c_y = curr_coord
            neighbours = [coord for coord in self.neighbours(curr_coord) if self.env.is_inside(coord)]
            areas = self.get_areas_fast2(s_map, curr_coord)
            for n_coord in neighbours:
                n_x, n_y = n_coord
                if not checked[n_y, n_x]:
                    checked[n_y, n_x] = True
                    if s_map[n_y, n_x] == self.env.FOOD_TILE:
                        # print('Found food')
                        food_count += 1
                    if self.is_single_area(n_coord, areas):
                        area_check = self.area_check(s_map, body_coords, n_coord, food_count=food_count, checked=checked)
                        if area_check['is_clear']:
                            return area_check
                        else:
                            # self_indexes.append(area_check['max_index'])
                            max_index = max(max_index, area_check['max_index'])
                            food_count += area_check['food_count']
                            tile_count += area_check['tile_count']
                    if coord_cmp(n_coord, tail_coord):
                        has_tail = True
                        is_clear = True
                        done = True
                        break
                    coord_val = s_map[n_y, n_x]
                    if coord_val == self.env.FREE_TILE or coord_val == self.env.FOOD_TILE:
                        tile_count += 1
                        current_coords.append(n_coord)
                    elif coord_val == self.body_value:
                        max_index = max(max_index, body_coords.index(n_coord))

            total_steps = tile_count - food_count
            needed_steps = body_len - max_index
            if total_steps >= needed_steps:
                is_clear = True
                break
            if done:
                break
            # self.print_map(self.show_search(s_map.copy(), checked, curr_coord, []))
            # print('max_index: ', max_index)
            # print('self_indexes: ', self_indexes)
            # print('curr_coord: ', curr_coord)
            # print('tile_count: ', tile_count)
            # print('total_steps: ', total_steps)
            # print('food_count: ', food_count)
            # print('needed_steps: ', needed_steps)
        result = {
                    'is_clear': is_clear,
                    'tile_count': tile_count,
                    'total_steps': total_steps,
                    'food_count': food_count,
                    'has_tail': has_tail,
                    'max_index': max_index,
                }
        # print(result)
        return result

    def is_area_clear(self, s_map, body_coords, start_coord, tile_count=1, food_count=0, checked=None):
        current_coords = [start_coord]
        safety_buffer = 0
        if checked is None:
            checked = np.full((self.env.height, self.env.width), fill_value=False, dtype=bool)
        body_len = len(body_coords)
        tail_coord = body_coords[-1]
        self_indexes = [0]
        # tile_count = 1
        # food_count = 0
        is_clear = False
        has_tail = False
        done = False
        while current_coords and not done:
            next_coords = []
            for curr_coord in current_coords:
                c_x, c_y = curr_coord
                checked[c_y, c_x] = True
                neighbours = [coord for coord in self.neighbours(curr_coord) if self.env.is_inside(coord)]
                unexplored_tiles = [(x, y) for (x, y) in neighbours if not checked[y, x]]
                # valid_tiles = self.valid_tiles(s_map, curr_coord)
                # if valid_tiles:
                #     areas = self.get_areas_fast(s_map, curr_coord, valid_tiles)
                # else:
                #     areas = []
                # print('curr_coord, ',curr_coord)
                # print('areas: ', areas)
                # print('unexplored_tiles: ', unexplored_tiles)
                areas = self.get_areas_fast2(s_map, curr_coord)
                for n_coord in unexplored_tiles:
                    t_x, t_y = n_coord
                    # If the tile is the tail, the area is clear
                    if s_map[t_y, t_x] == self.env.FOOD_TILE:
                        food_count += 1
                    if n_coord == tail_coord:
                        has_tail = True
                        is_clear = True
                        # done = True
                        # break
                    if s_map[t_y, t_x] in self.env.valid_tile_values:
                        checked[t_y, t_x] = True
                        tile_count += 1
                        if self.is_single_area(n_coord, areas):
                            # print('Checking subarea')
                            # print(n_coord)
                            area_check = self.is_area_clear(s_map, body_coords, n_coord, tile_count=tile_count, food_count=food_count, checked=checked)
                            if area_check['is_clear']:
                                return area_check
                        next_coords.append(n_coord)
                    elif s_map[t_y, t_x] == self.body_value:
                        self_indexes.append(body_coords.index(n_coord))
                # if done:
                #     break
            current_coords = next_coords
            total_steps = tile_count - food_count
            max_index = max(self_indexes)
            needed_steps = body_len - max_index + safety_buffer
            if total_steps >= needed_steps:
                is_clear = True
                # break
        result = {
                    'is_clear': is_clear,
                    'tile_count': tile_count,
                    'total_steps': total_steps,
                    'food_count': food_count,
                    'has_tail': has_tail
                }
        # print(result)
        return result


    def calc_immediate_risk(self, s_map, option_coord):
        ## OBS !!! OBS !!! OBS !!! OBS !!! OBS !!! OBS !!! OBS !!! OBS !!! OBS !!! OBS !!! OBS !!! OBS !!! OBS !!!
        return 0
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
                    s_map_copy = s_map.copy()
                    for op_option in comb:
                        head_val, coord = op_option
                        op_x, op_y = coord
                        s_map_copy[op_y][op_x] = head_val
                    next_states.append(s_map_copy)
            return next_states

        def recurse(s_map, self_coord, body_coords, self_length, depth=0):
            if not self.is_area_clear(s_map, body_coords, self_coord)['is_clear']:
                return 1
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


    def verify_route(self, route):
        if not route:
            return False
        if route[-1] not in self.valid_tiles(self.map, self.coord):
            return False
        for coord in route:
            x, y = coord
            if self.map[y, x] not in [self.env.FREE_TILE, self.env.FOOD_TILE, self.body_value, self.head_value]:
                return False
        return True

    def deep_look_ahead(self, s_map, new_coord, body_coords, length,
                        start_time=None,
                        planned_route=None,
                        old_route=None,
                        depth=1,
                        best_results=None,
                        current_results=None,
                        rundata=None,
                        failed_paths=None,
                        timeout_ms=None):
        if timeout_ms is None:
            timeout_ms = self.MAX_BRANCH_TIME
        if start_time is None:
            start_time = time()
        if failed_paths is None:
            failed_paths = set()
        planned_tile = None
        if current_results is None:
            current_results = {}
            current_results['timeout'] = False
            current_results['free_path'] = False
            current_results['apple_time'] = []
            current_results['len_gain'] = 0
            current_results['route'] = deque()
        if best_results is None:
            best_results = {}
        if s_map[new_coord[1], new_coord[0]] == self.env.FOOD_TILE:
            length += 1
            current_results['apple_time'] = current_results['apple_time'] + [depth]
            current_results['len_gain'] = length - self.length

        old_tail = self.update_body(new_coord, body_coords, length)
        s_map = self.update_snake_position(s_map, body_coords, old_tail)
        valid_tiles = self.valid_tiles(s_map, new_coord)
        if rundata is not None:
            rundata.append(body_coords.copy())

        current_results['body_coords'] = body_coords
        current_results['depth'] = depth
        current_results['route'] = current_results['route'].copy()
        current_results['route'].appendleft(new_coord)

        best_results['depth'] = max(best_results.get('depth', 0), current_results['depth'])
        best_results['len_gain'] = max(best_results.get('len_gain', 0), current_results['len_gain'])
        best_results['apple_time'] = []
        best_results['timeout'] = False
        best_results['free_path'] = False
        best_results['body_coords'] = max(best_results.get('body_coords', deque()), current_results['body_coords'].copy(), key=lambda o: len(o))
        best_results['route'] = max(best_results.get('route', deque()), current_results['route'], key=lambda o: len(o))


        if ((time() - start_time) * 1000 > timeout_ms) and self.TIME_LIMIT:
            best_results['timeout'] = True
            return best_results
        if current_results['depth'] >= length:
            current_results['free_path'] = True
            return current_results
        if planned_route:
            planned_tile = planned_route.pop()
            if planned_tile and planned_tile in valid_tiles:
                # print(f"Trying tile: {planned_tile}")
                check_result = self.deep_look_ahead(
                    s_map.copy(),
                    planned_tile,
                    body_coords.copy(),
                    length,
                    depth=depth+1,
                    planned_route=planned_route,
                    best_results=best_results,
                    old_route=old_route,
                    current_results=current_results.copy(),
                    start_time=start_time,
                    rundata=rundata,
                    failed_paths=failed_paths)
                if check_result['free_path'] or check_result['timeout']:
                    return check_result
                else:
                    planned_route = None
            # print(f"Planned tile {planned_tile} in {planned_route} failed")
        # if food_route := self.get_route(s_map, new_coord, target_tiles=self.env.food.locations):
        #     old_route = planned_route
        #     planned_route = food_route[:-1]
        if old_route:
            old_route_list = list(old_route)
            if new_coord in old_route:
                index = old_route.index(new_coord)
                planned_route = old_route_list[:index+1]
                old_route = None
            else:
                connecting_route = self.get_route(s_map, new_coord, target_tiles=old_route_list)
                if connecting_route:
                    index = old_route.index(connecting_route[0])
                    planned_route = old_route_list[:index] + connecting_route[:-1]

        if planned_route:
            target_tile = planned_route.pop()
            valid_tiles.sort(key=lambda x: 0 if x == target_tile else 1)
        else:
            # target_tile = None
            target_tile = self.target_tile(s_map, body_coords, recurse_mode=True)
        if valid_tiles:
            for tile in valid_tiles:
                state_tuple = tuple([self.get_flat_map_state(s_map), depth, tile])
                state_hash = hash(state_tuple)
                if state_hash in failed_paths:
                    continue
                c_time = time()
                # area_check = self.area_check(s_map, body_coords, tile)
                area_check = self.area_check(s_map, body_coords, tile)
                # print(f"Time for area check: {(time() - c_time) * 1000} ms")
                # print(f"Area check: {area_check}")
                if area_check['has_tail']:
                    current_results['free_path'] = True
                    current_results['len_gain'] = area_check['food_count']
                    current_results['depth'] = length
                    return current_results
                if not area_check['is_clear']:
                    best_results['depth'] = max(best_results['depth'], area_check['tile_count'])
                    continue
                if tile == target_tile and planned_route:
                    next_route = planned_route.copy()
                else:
                    next_route = None
                check_result = self.deep_look_ahead(
                    s_map.copy(),
                    tile,
                    body_coords.copy(),
                    length,
                    planned_route=next_route,
                    old_route=old_route,
                    depth=depth+1,
                    best_results=best_results,
                    current_results=current_results.copy(),
                    start_time=start_time,
                    rundata=rundata,
                    failed_paths=failed_paths)
                if check_result['free_path'] or check_result['timeout']:
                    return check_result
                failed_paths.add(state_hash)
        return best_results
