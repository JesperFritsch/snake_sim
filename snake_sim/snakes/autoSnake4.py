import random
import numpy as np
import itertools
from collections import deque
from statistics import mean
from time import time

from snake_sim.cpp_bindings.area_check import AreaChecker

from ..utils import coord_op, coord_cmp, exec_time

from .autoSnakeBase import AutoSnakeBase

class AutoSnake4(AutoSnakeBase):
    TIME_LIMIT = True
    MAX_RISK_CALC_DEPTH = 3

    def __init__(self, id: str, start_length: int, calc_timeout=1000, greedy=False):
        super().__init__(id, start_length)
        self.greedy = greedy
        self.calc_timeout = calc_timeout
        self.food_in_route = []
        self.failed_paths = set()
        self.area_checker = None

    def init_after_bind(self):
        self.area_checker = AreaChecker(
            self.env.FOOD_TILE,
            self.env.FREE_TILE,
            self.body_value,
            self.env.width,
            self.env.height)

    def fix_route(self, route, s_coord=None, valid_tiles=None):
        valid_tiles = valid_tiles or self.valid_tiles(self.map, self.coord)
        s_coord = s_coord or self.coord
        if s_coord in route and len(route) > 1:
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
        option['risk'] = 0
        # print("option tile: ", start_coord)
        # print('margin: ', option['margin'])
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
        else:
            planned_tile = self.target_tile(self.map, self.body_coords)
            planned_route = None
        valid_tiles.sort(key=lambda x: 0 if x == planned_tile else 1)
        for coord in valid_tiles:
            if coord == planned_tile:
                route = None
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
                    best_option = max(options.values(), key=lambda x: x['margin'])
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
        s_map = self.occupy_route(s_map, food_route)
        if self.area_check_wrapper(s_map, self.body_coords, end_coord)['is_clear']:
            return True
        return False

    def pick_direction(self):
        next_tile = None
        look_ahead_tile = None
        closest_food_route = self.get_route(self.map, self.coord, target_tiles=[l for l in self.env.food.locations if l != self.coord])
        if closest_food_route and self.check_safe_food_route(self.map.copy(), closest_food_route):
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
                    # print("to food")
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
        food_tile = self.env.FOOD_TILE
        free_tile = self.env.FREE_TILE
        blocked_tile = self.env.BLOCKED_TILE
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
                if s_map[n_coord[1], n_coord[0]] == free_tile or s_map[n_coord[1], n_coord[0]] == food_tile:
                    curr_area.append(n_coord)
                    curr_area_len += 1
                elif curr_area_len != 0:
                    areas.append(curr_area)
                    curr_area = deque()
                    curr_area_len = 0
                corner_coord = corner_coords[i]
                if (self.env.is_inside(corner_coord) and not
                    (s_map[corner_coord[1], corner_coord[0]] == free_tile or s_map[corner_coord[1], corner_coord[0]] == food_tile) and
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
        if areas[0][0] == n_coords[0] and areas[-1][-1] == n_coords[-1] and curr_area_len != 4:
            lu_corn = corner_coords[-1]
            if s_map[lu_corn[1], lu_corn[0]] == free_tile or s_map[lu_corn[1], lu_corn[0]] == food_tile:
                areas[0].extend(areas.pop())
        return areas


    def is_single_area(self, coord, areas):
        for area in areas:
            if len(area) == 1 and (area[0][0] == coord[0] and area[0][1] == coord[1]):
                return True
        return False


    def is_area_entrance(self, s_map, coord, check_coord):
        """
        check if a coordinate leads in to a different area

        F = coord
        T = check_coord
        number = check order

        1 T 1
        2 F 2

        """
        food_value = self.env.FOOD_TILE
        free_value = self.env.FREE_TILE
        c_perp = (check_coord[1] - coord[1], check_coord[0] - coord[0])
        x = check_coord[0] + c_perp[0]
        y = check_coord[1] + c_perp[1]
        if (0 <= x < self.width and 0 <= y < self.height) and (s_map[y, x] == free_value or s_map[y, x] == food_value):
            x = coord[0] + c_perp[0]
            y = coord[1] + c_perp[1]
            if (0 <= x < self.width and 0 <= y < self.height) and (s_map[y, x] == free_value or s_map[y, x] == food_value):
                return False
        x = check_coord[0] - c_perp[0]
        y = check_coord[1] - c_perp[1]
        if (0 <= x < self.width and 0 <= y < self.height) and (s_map[y, x] == free_value or s_map[y, x] == food_value):
            x = coord[0] - c_perp[0]
            y = coord[1] - c_perp[1]
            if (0 <= x < self.width and 0 <= y < self.height) and (s_map[y, x] == free_value or s_map[y, x] == food_value):
                return False
        return True


    def area_check_needed(self, s_map, head_coord, neck_coord):
        """
        This should cover all scenarios where an area check is needed.
        checking if the head will close of any area.
        and checking if the head creates paths that are one tile wide.
        """
        free_value = self.env.FREE_TILE
        food_value = self.env.FOOD_TILE
        head_dir = (head_coord[0] - neck_coord[0], head_coord[1] - neck_coord[1])
        perp_axis = (head_dir[1], head_dir[0])
        coord_ahead = (head_coord[0] + head_dir[0], head_coord[1] + head_dir[1])

        if not self.env.is_inside(coord_ahead):
            return True

        if s_map[coord_ahead[1], coord_ahead[0]] != free_value and s_map[coord_ahead[1], coord_ahead[0]] != food_value:
            return True
        besides_a = (head_coord[0] + perp_axis[0], head_coord[1] + perp_axis[1])
        besides_b = (head_coord[0] - perp_axis[0], head_coord[1] - perp_axis[1])
        diag_a_ahead = (coord_ahead[0] + perp_axis[0], coord_ahead[1] + perp_axis[1])
        diag_b_ahead = (coord_ahead[0] - perp_axis[0], coord_ahead[1] - perp_axis[1])

        if self.env.is_inside(diag_a_ahead) and self.env.is_inside(besides_a):
            if s_map[diag_a_ahead[1], diag_a_ahead[0]] != free_value and s_map[diag_a_ahead[1], diag_a_ahead[0]] != food_value:
                if s_map[besides_a[1], besides_a[0]] == free_value or s_map[besides_a[1], besides_a[0]] == food_value:
                    return True
            besides2 = (head_coord[0] + perp_axis[0] * 2, head_coord[1] + perp_axis[1] * 2)
            if s_map[besides_a[1], besides_a[0]] == free_value or s_map[besides_a[1], besides_a[0]] == food_value:
                if self.env.is_inside(besides2) and (s_map[besides2[1], besides2[0]] != free_value and s_map[besides2[1], besides2[0]] != food_value):
                    return True

        if self.env.is_inside(diag_b_ahead) and self.env.is_inside(besides_b):
            if s_map[diag_b_ahead[1], diag_b_ahead[0]] != free_value and s_map[diag_b_ahead[1], diag_b_ahead[0]] != food_value:
                if s_map[besides_b[1], besides_b[0]] == free_value or s_map[besides_b[1], besides_b[0]] == food_value:
                    return True
            besides2 = (head_coord[0] - perp_axis[0] * 2, head_coord[1] - perp_axis[1] * 2)
            if s_map[besides_b[1], besides_b[0]] == free_value or s_map[besides_b[1], besides_b[0]] == food_value:
                if self.env.is_inside(besides2) and (s_map[besides2[1], besides2[0]] != free_value and s_map[besides2[1], besides2[0]] != food_value):
                    return True
        return False


    def area_check_wrapper(self, s_map, body_coords, start_coord):
        # return self.area_check(s_map, body_coords, start_coord)
        return self.area_checker.area_check(s_map, list(body_coords), start_coord)

    def area_check(self, s_map, body_coords, start_coord, tile_count=0, food_count=0, max_index=0, checked=None, depth=0):
        current_coords = deque([start_coord])
        to_be_checked = deque()
        if checked is None:
            checked = np.full((self.env.height, self.env.width), fill_value=False, dtype=bool)
        checked[start_coord[1], start_coord[0]] = True
        body_len = len(body_coords)
        tail_coord = body_coords[-1]
        is_clear = False
        has_tail = False
        done = False
        total_steps = 0
        tile_count += 1
        food_value = self.env.FOOD_TILE
        free_value = self.env.FREE_TILE
        body_value = self.body_value
        while current_coords:
            curr_coord = current_coords.popleft()
            c_x, c_y = curr_coord
            if s_map[c_y, c_x] == food_value:
                food_count += 1
            neighbours = ((c_x, c_y-1),(c_x+1, c_y),(c_x, c_y+1),(c_x-1, c_y))
            for n_coord in neighbours:
                n_x, n_y = n_coord
                if (0 <= n_x < self.width and 0 <= n_y < self.height):
                    if not checked[n_y, n_x]:
                        checked[n_y, n_x] = True
                        coord_val = s_map[n_y, n_x]
                        if coord_val == free_value or coord_val == food_value:
                            if self.is_area_entrance(s_map, curr_coord, n_coord):
                                to_be_checked.append(n_coord)
                                continue
                            tile_count += 1
                            current_coords.append(n_coord)
                        elif coord_val == body_value:
                            body_index = body_coords.index(n_coord)
                            if body_index > max_index:
                                max_index = body_index
                        if n_coord[0] == tail_coord[0] and n_coord[1] == tail_coord[1] and not (tile_count == food_count == 1):
                            has_tail = True
                            is_clear = True
                            done = True
                            break
            total_steps = tile_count - food_count
            needed_steps = body_len - max_index
            if total_steps >= needed_steps:
                is_clear = True
                # break
            if done:
                break
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
        result = {
                    'is_clear': is_clear,
                    'tile_count': tile_count,
                    'total_steps': total_steps,
                    'food_count': food_count,
                    'has_tail': has_tail,
                    'max_index': max_index,
                    'start_coord': start_coord,
                    'needed_steps': body_len - max_index
                }
        # print(result)
        # print('depth: ', depth)
        return result


    def calc_immediate_risk(self, s_map, option_coord, max_depth=None):
        return 0
        if max_depth is None:
            max_depth = self.MAX_RISK_CALC_DEPTH
        ## OBS !!! OBS !!! OBS !!! OBS !!! OBS !!! OBS !!! OBS !!! OBS !!! OBS !!! OBS !!! OBS !!! OBS !!! OBS !!!
        def coords_unique(tuples):
            coords = [coord for _, coord in tuples]
            return len(set(coords)) == len(coords)

        def get_next_states(s_map):
            op_valid_tiles = {}
            for snake in [s for s in self.env.snakes.values() if s.alive and s is not self]:
                coord_lists = np.where(s_map == snake.head_value, s_map)
                h_coord = (coord_lists[1][0], coord_lists[0][0])
                op_valids = [c for c in self.valid_tiles(s_map, h_coord)]
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
                    combinations = list({tuple(sorted(combination)) for combination in combinations})
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
            if depth >= max_depth:
                return 0
            results = []
            if s_map[self_coord[1], self_coord[0]] == self.env.FOOD_TILE:
                self_length += 1
            old_tail = self.update_body(self_coord, body_coords, self_length)
            for next_state_map in get_next_states(s_map.copy()):
                if valids_in_next := self.valid_tiles(next_state_map, self_coord):
                    sub_results = []
                    for self_valid in valids_in_next:
                        next_recurse_map = self.update_snake_position(next_state_map.copy(), body_coords, old_tail)
                        # self.print_map(next_recurse_map)
                        # print('depth: ', depth)
                        # print(self.id)
                        result = recurse(next_recurse_map, self_valid, body_coords.copy(), self_length, depth+1)
                        sub_results.append(result)
                    results.append(mean(sub_results))
                else:
                    results.append(1)
            if results:
                return mean(results)
            else:
                return 0

        self_length = self.length
        body_coords = self.body_coords.copy()
        old_tail = self.update_body(option_coord, body_coords, self_length)
        next_state_map = self.update_snake_position(s_map.copy(), body_coords, old_tail)
        return recurse(next_state_map, option_coord, body_coords, self_length)


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
                        rundata=None,
                        timeout_ms=None,
                        check_areas=True,
                        area_check_data=None):
        safety_buffer = 3
        if timeout_ms is None:
            timeout_ms = self.calc_timeout
        if start_time is None:
            start_time = time()
        if area_check_data is None:
            area_check_data = {}
        planned_tile = None
        if current_results is None:
            current_results = {}
            current_results['timeout'] = False
            current_results['free_path'] = False
            current_results['apple_time'] = []
            current_results['len_gain'] = 0
            current_results['route'] = deque()
            current_results['margin'] = -length
        if best_results is None:
            best_results = {}
        if s_map[new_coord[1], new_coord[0]] == self.env.FOOD_TILE:
            length += 1
            current_results['apple_time'] = current_results['apple_time'] + [depth]
            current_results['len_gain'] = length - self.length
        current_results['body_coords'] = body_coords
        current_results['depth'] = depth
        current_results['route'] = current_results['route'].copy()
        current_results['route'].appendleft(new_coord)

        best_results['depth'] = max(best_results.get('depth', 0), current_results['depth'])
        best_results['len_gain'] = max(best_results.get('len_gain', 0), current_results['len_gain'])
        best_results['margin'] = max(best_results.get('margin', -length), current_results['margin'])
        best_results['apple_time'] = []
        best_results['timeout'] = False
        best_results['free_path'] = False
        best_results['body_coords'] = max(best_results.get('body_coords', deque()), current_results['body_coords'].copy(), key=lambda o: len(o))
        best_results['route'] = max(best_results.get('route', deque()), current_results['route'], key=lambda o: len(o))
        old_tail = self.update_body(new_coord, body_coords, length)
        s_map = self.update_snake_position(s_map, body_coords, old_tail)
        valid_tiles = self.valid_tiles(s_map, new_coord)
        if rundata is not None:
            rundata.append(body_coords.copy())



        if ((time() - start_time) * 1000 > timeout_ms) and self.TIME_LIMIT:
            best_results['timeout'] = True
            return best_results

        if current_results['depth'] >= length + safety_buffer:
            current_results['free_path'] = True
            return current_results

        # needed_steps = area_check_data.get('needed_steps', 1) - 1
        # if needed_steps < 0:
        #     return current_results

        area_checks = {}

        if planned_route:
            planned_tile = planned_route.pop()
            if planned_tile and planned_tile in valid_tiles:
                area_check = self.area_check_wrapper(s_map, body_coords, planned_tile)
                area_checks[planned_tile] = area_check
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
                    rundata=rundata)
                if check_result['free_path'] or check_result['timeout']:
                    return check_result
                else:
                    planned_route = None


        if valid_tiles:
            best_margin = -length
            target_tile = None
            for tile in valid_tiles:
                area_check = area_checks.get(tile, None)
                if area_check is None:
                    area_check = self.area_check_wrapper(s_map, body_coords, tile)
                    area_checks[tile] = area_check
                # print('tile: ', tile)
                # print('area_check: ', area_check)
                if area_check['margin'] > best_margin:
                    best_margin = area_check['margin']
                    best_results['margin'] = max(best_results['margin'], best_margin)
                    target_tile = tile
            # if the best margin here is less than the best margin from the previous iteration
            # it means we but of an area, and that could lead to difficulties finding a path.
            # best_margin + 3 was needed, otherwise we return even when its fine, for some reason.
            # print('best margin: ', best_margin)
            # print('current margin: ', current_results['margin'])
            # self.print_map(s_map)
            if (best_margin + 3) < current_results['margin'] and best_margin < length:
                # print('margin break')
                return best_results
            if target_tile is None:
                target_tile = self.target_tile(s_map, body_coords)
            valid_tiles.sort(key=lambda x: 0 if x == target_tile else 1)

            for tile in valid_tiles:
                state_tuple = tuple([self.get_flat_map_state(s_map), tile])
                state_hash = hash(state_tuple)
                if state_hash in self.failed_paths:
                    continue
                area_check = area_checks[tile].copy()
                # print(area_check)
                # self.print_map(s_map)
                if area_check['has_tail']:
                    current_results['free_path'] = True
                    current_results['len_gain'] = area_check['food_count']
                    current_results['depth'] = length
                    return current_results
                if not area_check['is_clear']:
                    best_results['depth'] = max(best_results['depth'], area_check['tile_count'])
                    best_results['margin'] = max(best_results['margin'], current_results['margin'])
                    continue
                current_results['margin'] = area_check['margin']

                check_result = self.deep_look_ahead(
                    s_map.copy(),
                    tile,
                    body_coords.copy(),
                    length,
                    old_route=old_route,
                    depth=depth+1,
                    best_results=best_results,
                    current_results=current_results.copy(),
                    start_time=start_time,
                    rundata=rundata,
                    area_check_data=area_check_data)
                if check_result['free_path'] or check_result['timeout']:
                    return check_result
                self.failed_paths.add(state_hash)
        return best_results
