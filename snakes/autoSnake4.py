import random
import numpy as np
import itertools
from collections import deque
from statistics import mean
from time import time

from utils import coord_op, coord_cmp, exec_time

from snakes.autoSnakeBase import AutoSnakeBase

class AutoSnake4(AutoSnakeBase):
    TIME_LIMIT = True
    MAX_RISK_CALC_DEPTH = 3

    def __init__(self, id: str, start_length: int, calc_timeout=1000, greedy=False):
        super().__init__(id, start_length)
        self.greedy = greedy
        self.calc_timeout = calc_timeout
        self.food_in_route = []
        self.failed_paths = set()

    def fix_route(self, route, s_coord=None, valid_tiles=None):
        valid_tiles = valid_tiles or self.valid_tiles(self.map, self.coord)
        s_coord = s_coord or self.coord
        if s_coord in route:
            route = deque(list(route)[:route.index(s_coord)])
        try:
            if route[-1] not in valid_tiles:
                try:
                    sub_route = self.get_route(self.map, s_coord, end=route[-1])
                    route = deque(list(route) + sub_route[1:-1])
                except Exception as e:
                    print(e)
                    raise ValueError('Invalid route')
        except IndexError as e:
            print(e)
            print('Route: ', route)
        except Exception as e:
            print(e)
            print(route)
        return route

    def set_route(self, route: deque):
        if not route:
            self.route = None
            return
        route = self.fix_route(route)
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
        planned_tile = None
        if self.route:
            planned_route = self.route.copy()
            planned_tile = planned_route.pop()
            valid_tiles.sort(key=lambda x: 0 if x == planned_tile else 1)
        for coord in valid_tiles:
            if coord == planned_tile:
                route = planned_route
            else:
                route = None
            option = self.find_route(coord, planned_route=route)
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
            option = self.find_route(look_ahead_tile, planned_route=planned_route, old_route=old_route)
            if option['free_path']:
                # print(option['route'])
                # print(self.coord)
                self.set_route(option['route'])
                if self.route:
                    next_tile = self.route.pop()
                    return next_tile

        route = self.get_best_route()
        # print(self.coord)
        self.set_route(route)
        # print('route: ', self.route)
        if self.route:
            next_tile = self.route.pop()
            if not len(self.route) >= self.length:
                self.route = None
        else:
            next_tile = None
        return next_tile

    def target_tile(self, s_map, body_coords, recurse_mode=False):
        self_coord = body_coords[0]
        if not recurse_mode and (attack_moves := self.find_attack_moves(s_map)):
            return coord_op(self_coord, attack_moves[0], '+')
        s_dir = coord_op(self_coord, body_coords[1], '-')
        return coord_op(self_coord, s_dir, '+')

    def get_areas_fast(self, s_map, s_coord):
        curr_area = deque()
        areas = []
        s_x, s_y = s_coord
        n_coords = ((s_x, s_y-1),(s_x+1, s_y),(s_x, s_y+1),(s_x-1, s_y))
        corner_coords = ((s_x+1, s_y-1), (s_x+1, s_y+1), (s_x-1, s_y+1), (s_x-1, s_y-1))
        i = -1
        curr_area_len = 0
        for n_coord in n_coords:
            i += 1
            if self.env.is_inside(n_coord):
                if s_map[n_coord[1], n_coord[0]] == self.env.FREE_TILE or s_map[n_coord[1], n_coord[0]] == self.env.FOOD_TILE:
                    curr_area.append(n_coord)
                    curr_area_len += 1
                elif curr_area_len != 0:
                    areas.append(curr_area)
                    curr_area = deque()
                    curr_area_len = 0
                corner_coord = corner_coords[i]
                if (self.env.is_inside(corner_coord) and
                    s_map[corner_coord[1], corner_coord[0]] != self.env.FREE_TILE and s_map[corner_coord[1], corner_coord[0]] != self.env.FOOD_TILE and
                    curr_area_len != 0):
                        areas.append(curr_area)
                        curr_area = deque()
                        curr_area_len = 0
            elif curr_area_len != 0:
                areas.append(curr_area)
                curr_area = deque()
                curr_area_len = 0
        if curr_area_len != 0:
            areas.append(curr_area)
        #if the last area is connected to the first area, merge them
        if areas[0][0] == n_coords[0] and areas[-1][-1] == n_coords[-1]:
            lu_corn = corner_coords[-1]
            if s_map[lu_corn[1], lu_corn[0]] == self.env.FREE_TILE or s_map[lu_corn[1], lu_corn[0]] == self.env.FOOD_TILE:
                areas[0].extend(areas.pop())
        return areas


    def is_single_area(self, coord, areas):
        for area in areas:
            if len(area) == 1 and coord_cmp(area[0], coord):
                return True
        return False

    def area_check(self, s_map, body_coords, start_coord, tile_count=0, food_count=0, max_index=0, checked=None, depth=0):
        current_coords = deque([start_coord])
        to_be_checked = deque()
        if checked is None:
            checked = np.full((self.env.height, self.env.width), fill_value=False, dtype=bool)
        body_len = len(body_coords)
        tail_coord = body_coords[-1]
        # self_indexes = [0]
        is_clear = False
        has_tail = False
        done = False
        total_steps = 0
        tile_count += 1
        if s_map[start_coord[1], start_coord[0]] == self.env.FOOD_TILE:
            food_count += 1
        checked[start_coord[1], start_coord[0]] = True
        while current_coords:
            curr_coord = current_coords.popleft()
            c_x, c_y = curr_coord
            if tile_count > 1:
                #if it is the first tile then an area search could be incorrect if the head is one step away from a
                areas = self.get_areas_fast(s_map, curr_coord)
            else:
                areas = []
            # map_copy = self.show_search(s_map.copy(), coord=curr_coord, checked=checked)
            # self.print_map(map_copy)
            # print('areas: ', areas)
            neighbours = ((c_x, c_y-1),(c_x+1, c_y),(c_x, c_y+1),(c_x-1, c_y))
            neighbours = [coord for coord in neighbours if self.env.is_inside(coord)]
            for n_coord in neighbours:
                n_x, n_y = n_coord
                if not checked[n_y, n_x]:
                    checked[n_y, n_x] = True
                    if coord_cmp(n_coord, tail_coord):
                        has_tail = True
                        is_clear = True
                        done = True
                        break
                    coord_val = s_map[n_y, n_x]
                    if self.is_single_area(n_coord, areas):
                        to_be_checked.append(n_coord)
                        continue
                    if coord_val == self.env.FREE_TILE or coord_val == self.env.FOOD_TILE:
                        if s_map[n_y, n_x] == self.env.FOOD_TILE:
                            # print('Found food')
                            food_count += 1
                        tile_count += 1
                        current_coords.append(n_coord)
                    elif coord_val == self.body_value:
                        body_index = body_coords.index(n_coord)
                        if body_index > max_index:
                            max_index = body_index
            total_steps = tile_count - food_count
            needed_steps = body_len - max_index
            if total_steps >= needed_steps:
                is_clear = True
                # break
            if done:
                break
        # print('AT THE END!!!!!!')
        # print('total_steps: ', total_steps)
        # print('needed_steps: ', needed_steps)
        # print('body_len: ', body_len)
        # print('max_index: ', max_index)
        # print('food_count: ', food_count)
        # print('AT THE END!!!!!!')
        if not is_clear:
            while to_be_checked:
                coord = to_be_checked.popleft()
                area_check = self.area_check(
                    s_map,
                    body_coords,
                    coord,
                    tile_count=tile_count,
                    food_count=food_count,
                    checked=checked,
                    depth=depth+1)
                if area_check['is_clear']:
                    return area_check
                # else:
                #     check_food_count = food_count + area_check['food_count']
                #     check_tile_count = tile_count + area_check['tile_count']
                #     total_steps = check_tile_count - check_food_count
                #     needed_steps = body_len - area_check['max_index']
                #     max_index = area_check['max_index']
                #     print('total_steps: ', total_steps)
                #     print('needed_steps: ', needed_steps)
                #     print('body_len: ', body_len)
                #     print('max_index: ', area_check['max_index'])
                #     print('food_count: ', food_count)
                #     if total_steps >= needed_steps:
                #         food_count = check_food_count
                #         tile_count = check_tile_count
                #         is_clear = True
                #         break

        result = {
                    'is_clear': is_clear,
                    'tile_count': tile_count,
                    'total_steps': total_steps,
                    'food_count': food_count,
                    'has_tail': has_tail,
                    'max_index': max_index,
                    'start_coord': start_coord,
                }
        # print(result)
        # print('depth: ', depth)
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
                        old_route=None,
                        depth=1,
                        best_results=None,
                        planned_route=None,
                        current_results=None,
                        failed_paths=None,
                        rundata=None,
                        timeout_ms=None):
        safety_buffer = 3
        # if failed_paths is None:
        #     failed_paths = set()
        if timeout_ms is None:
            timeout_ms = self.calc_timeout
        if start_time is None:
            start_time = time()
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
        if current_results['depth'] >= length + safety_buffer:
            current_results['free_path'] = True
            return current_results

        area_checks = {}

        if planned_route:
            planned_tile = planned_route.pop()
            if planned_tile and planned_tile in valid_tiles:
                valid_tiles.remove(planned_tile)
                area_check = self.area_check(s_map, body_coords, planned_tile)
                area_checks[planned_tile] = area_check
                # print(tile, area_check)
                if area_check['has_tail']:
                    current_results['free_path'] = True
                    current_results['len_gain'] = area_check['food_count']
                    current_results['depth'] = length
                    return current_results
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
        # if old_route:
        #     old_route_list = list(old_route)
        #     if new_coord in old_route:
        #         index = old_route.index(new_coord)
        #         planned_route = old_route_list[:index+1]
        #         old_route = None
        #     else:
        #         connecting_route = self.get_route(s_map, new_coord, target_tiles=old_route_list)
        #         if connecting_route:
        #             index = old_route.index(connecting_route[0])
        #             planned_route = old_route_list[:index] + connecting_route[:-1]

        # if planned_route:
        #     target_tile = planned_route.pop()
        #     valid_tiles.sort(key=lambda x: 0 if x == target_tile else 1)
        # else:
        #     target_tile = self.target_tile(s_map, body_coords, recurse_mode=True)
        # self.print_map(s_map)

        if valid_tiles:
            for tile in valid_tiles:
                state_tuple = tuple([self.get_flat_map_state(s_map), tile])
                state_hash = hash(state_tuple)
                if state_hash in self.failed_paths:
                    continue
                # c_time = time()
                # map_copy = self.show_search(s_map.copy(), coord=tile)
                if (area_check := area_checks.get(tile, None)) is None:
                    area_check = self.area_check(s_map, body_coords, tile)
                    area_checks[tile] = area_check
                # print(tile, area_check)
                if area_check['has_tail']:
                    current_results['free_path'] = True
                    current_results['len_gain'] = area_check['food_count']
                    current_results['depth'] = length
                    return current_results

                # area_check = self.area_check(s_map, body_coords, tile)
                if not area_check['is_clear']:
                    best_results['depth'] = max(best_results['depth'], area_check['tile_count'])
                    # best_results['free_path'] = False
                    continue
                # if tile == target_tile and planned_route:
                #     next_route = planned_route.copy()
                # else:
                #     next_route = None
                check_result = self.deep_look_ahead(
                    s_map.copy(),
                    tile,
                    body_coords.copy(),
                    length,
                    # planned_route=next_route,
                    old_route=old_route,
                    depth=depth+1,
                    best_results=best_results,
                    current_results=current_results.copy(),
                    start_time=start_time,
                    rundata=rundata,
                    failed_paths=failed_paths)
                if check_result['free_path'] or check_result['timeout']:
                    return check_result
                self.failed_paths.add(state_hash)
        return best_results
