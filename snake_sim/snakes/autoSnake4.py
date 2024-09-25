import random
import numpy as np
import itertools
from collections import deque
from statistics import mean
from functools import cmp_to_key
from time import time
from pprint import pprint

from snake_sim.cpp_bindings.area_check import AreaChecker

from ..utils import coord_op, distance, exec_time

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
                    print("Coord: ", s_coord)
                    raise ValueError('Invalid route')
        except IndexError as e:
            print(e)
            print('Route: ', route)
            print("Coord: ", s_coord)
        except Exception as e:
            print(e)
            print('Route: ', route)
            print("Coord: ", s_coord)
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

    def explore_option(self, start_coord, food_ahead=None, old_route=None, timeout_ms=None):
        time_s = time()
        branch_common = None
        if food_ahead is not None:
            branch_common = {}
            branch_common['min_margin'] = food_ahead
        option = self.deep_look_ahead(
                self.map.copy(),
                start_coord,
                self.body_coords.copy(),
                self.length,
                start_time=time_s,
                old_route=old_route,
                timeout_ms=timeout_ms,
                branch_common=branch_common)
        option['coord'] = start_coord
        option['risk'] = 0
        # print("option tile: ", start_coord)
        # print('margin: ', option['margin'])
        # print(option)
        return option


    def get_best_option(self):
        options = {}
        best_option = None
        valid_tiles = self.valid_tiles(self.map, self.coord)
        area_checks = {}
        target_tile = None
        best_margin = -self.length
        if not valid_tiles:
            return None
        for tile in valid_tiles:
            area_checks[tile] = self.area_check_wrapper(self.map, self.body_coords, tile, exhaustive=True)
            if area_checks[tile]['margin'] > best_margin:
                best_margin = area_checks[tile]['margin']
                target_tile = tile
        valid_tiles.sort(key=lambda x: 0 if x == target_tile else 1)
        for coord in valid_tiles:
            option = self.explore_option(coord)
            options[coord] = option
            # print('found option: ', option)
            # print('coord: ', coord)
            if option['free_path'] and area_checks[coord]['margin'] >= area_checks[coord]['food_count']:
                break
        # print(area_checks)
        free_options = [o for o in options.values() if o['free_path']]
        if free_options:
            free_coords = [o['coord'] for o in free_options]
            best_option_coord = max(free_coords, key=lambda x: area_checks[x]['margin'])
            best_option = options[best_option_coord]
        else:
            best_margin_area_tile = max(area_checks.items(), key=lambda x: x[1]['margin'])[0]
            best_option = options[best_margin_area_tile]
        if best_option is not None:
            return best_option
        else:
            return None

    def head_in_open(self):
        neighbours = ((-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0))
        head_dir = coord_op(self.body_coords[1], self.coord, '-')
        for n in neighbours:
            if n == head_dir:
                continue
            x, y = coord_op(self.coord, n, '+')
            if not self.env.is_inside((x, y) or self.map[y, x] > self.env.FREE_TILE):
                return False
        return True

    def check_safe_food_route(self, s_map, food_route):
        end_coord = food_route[0]
        body_copy = self.body_coords.copy()
        map_copy = s_map.copy()
        for coord in food_route:
            old_tail = self.update_body(coord, body_copy, self.length)
            self.update_snake_position(map_copy, body_copy, old_tail)
        valid_tiles = self.valid_tiles(s_map, end_coord)
        area_checks = [self.area_check_wrapper(s_map, body_copy, tile) for tile in valid_tiles]
        return any([a["margin"] >= a["food_count"] for a in area_checks])

    def get_closest_accessible_food_route(self):
        s_map = self.map.copy()
        food_locations = self.env.food.locations.copy()
        route = None
        while route := self.get_route(self.map, self.coord, target_tiles=[l for l in food_locations if l != self.coord]):
            if self.check_safe_food_route(s_map, route):
                return route
            else:
                food_locations.remove(route[0])
        return route

    def get_future_available_food_map(self):
        s_map = self.map.copy()
        valid_tiles = self.valid_tiles(self.map, self.coord)
        future_valids = {coord: self.valid_tiles(self.map, coord) for coord in valid_tiles}
        food_map = {}
        all_area_checks = {}
        all_clear_checks = {}
        additonal_food = {}
        best_checks = []
        for coord, valids in future_valids.items():
            if not valids:
                continue
            x, y = coord
            old_map_value = s_map[y, x]
            s_map[y, x] = self.env.BLOCKED_TILE
            area_checks = [self.area_check_wrapper(s_map, self.body_coords, tile, food_check=True) for tile in valids]
            s_map[y, x] = old_map_value
            clear_checks = [a for a in area_checks if a['is_clear']]
            # print(coord, clear_checks)
            all_area_checks[coord] = area_checks
            if clear_checks:
                all_clear_checks[coord] = clear_checks
            additonal_food[coord] = old_map_value == self.env.FOOD_TILE
        # all_checks = [a for check in all_area_checks.values() for a in check]
        # combine_food = all([a['margin'] >= a['food_count'] and a["food_count"] > 0 for a in all_checks])
        # pprint(all_area_checks)
        # print("combine food: ", combine_food)
        # combined_food = {}
        # for a in [a for a in all_checks]:
        #     combined_food.update(a['food_coords'])
        for coord, area_checks in all_clear_checks.items():
            # print("coord: ", coord)
            # print("area checks: ", area_checks)
            # if combine_food:
            #     most_food = len(combined_food)
            # else:
            most_food = max(area_checks, key=lambda x: x['food_count'])['food_count']
            # print("most food: ", most_food)
            # print("additonal food: ", additonal_food)
            food_map[coord] = most_food + (1 if additonal_food[coord] else 0)
        return food_map

    def get_available_areas(self):
        s_map = self.map.copy()
        valid_tiles = self.valid_tiles(self.map, self.coord)
        areas_map = {coord: self.area_check_wrapper(s_map, self.body_coords, coord) for coord in valid_tiles}
        return areas_map

    def pick_direction(self):
        next_tile = None
        planned_tile = None
        planned_route = None
        closest_food_route = self.get_closest_accessible_food_route()
        if closest_food_route:
            planned_route = closest_food_route[:-1]
            planned_tile = planned_route.pop()
        areas_map = self.get_available_areas()
        food_map = self.get_future_available_food_map()
        # print("areas map: ", areas_map)
        # print("food map: ", food_map)
        # print("food route: ", closest_food_route)
        # print("planned tile: ", planned_tile)
        if food_map and planned_tile is not None:
            best_food_pair = max(food_map.items(), key=lambda x: x[1])
            max_food_value = max(food_map.values())
            best_food_pairs = [pair for pair in food_map.items() if pair[1] == max_food_value]
            def cmp(c1, c2):
                return distance(planned_tile, c1[0]) - distance(planned_tile, c2[0])
            best_food_pair = min(best_food_pairs, key=cmp_to_key(cmp))
            # print("best food pairs: ", best_food_pairs)
            # print("best food pair: ", best_food_pair)
            best_food_tile, best_food_value = best_food_pair
            planned_tile_food_value = food_map.get(planned_tile, 0)
            # print(planned_tile is None, (planned_tile_food_value < best_food_value, not self.head_in_open()))
            if planned_tile is None or planned_tile_food_value < best_food_value:
                # print("food route is not best")
                planned_tile = best_food_tile
            planned_area = areas_map[planned_tile]
            #this is to make sure that spawning food wont kill us
            # print("margin: ", planned_area['margin'], "food:", planned_area['food_count'])
            if planned_area['margin'] >= planned_area['food_count']:
                # print("margin enough")
                # print("planned_area: ", planned_area)
                option = self.explore_option(planned_tile, food_ahead=planned_area['food_count'])
                if option['free_path']:
                    # print("free path")
                    # print("option: ", option)
                    next_tile = planned_tile
        if next_tile is None:
            # print("getting best route")
            option = self.get_best_option()
            if option:
                next_tile = option['coord']
        # print("next_tile: ", next_tile)
        # self.print_map(self.map)
        return next_tile

    def pick_direction_old(self):
        next_tile = None
        planned_tile = None
        closest_food_route = self.get_route(self.map, self.coord, target_tiles=[l for l in self.env.food.locations if l != self.coord])
        if closest_food_route and self.check_safe_food_route(self.map.copy(), closest_food_route):
            planned_route = closest_food_route[:-1]
            old_route = self.route
        else:
            old_route = None
            planned_route = self.route
        food_map = self.get_available_food_map()
        if food_map:
            best_food_pair = max(food_map.items(), key=lambda x: x[1])
            best_food_tile, best_food_value = best_food_pair
            # self.print_map(self.map)
            # print('best_food_tile: ', best_food_tile)
            # print('best_food_value: ', best_food_value)
            # print('food_map: ', food_map)
            if self.verify_route(planned_route):
                planned_tile = planned_route.pop()
                # print("planned_tile: ", planned_tile)
                if food_map and planned_tile in food_map and food_map[planned_tile] < best_food_value:
                    option = self.explore_option(best_food_tile, planned_route=planned_route, old_route=old_route)
                else:
                    option = self.explore_option(planned_tile, planned_route=planned_route, old_route=old_route)
                if option['free_path']:
                    self.set_route(option['route'])
                    if self.route:
                        next_tile = self.route.pop()
                        return next_tile
            else:
                option = self.explore_option(best_food_tile)
                if option['free_path']:
                    self.set_route(option['route'])
                    if self.route:
                        next_tile = self.route.pop()
                        return next_tile
        route = self.get_best_route()
        self.set_route(route)
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


    def area_check_wrapper(self, s_map, body_coords, start_coord, target_margin=0, food_check=False, exhaustive=False):
        # return self.area_check(s_map, body_coords, start_coord)
        return self.area_checker.area_check(s_map, list(body_coords), start_coord, target_margin, food_check, exhaustive)

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
        if max_depth is None:
            max_depth = self.MAX_RISK_CALC_DEPTH
        ## OBS !!! OBS !!! OBS !!! OBS !!! OBS !!! OBS !!! OBS !!! OBS !!! OBS !!! OBS !!! OBS !!! OBS !!! OBS !!!
        def coords_unique(tuples):
            coords = [coord for _, coord in tuples]
            return len(set(coords)) == len(coords)

        def get_next_states(s_map):
            op_valid_tiles = {}
            # print(self.env.snakes)
            for snake in [s for s in self.env.snakes.values() if s.alive and s is not self]:
                # print(snake.head_value)
                coord_lists = np.where(s_map == snake.head_value)
                # print('coord_lists: ', coord_lists)
                h_coord = (coord_lists[1][0], coord_lists[0][0])
                # print('h_coord: ', h_coord)
                op_valids = [c for c in self.valid_tiles(s_map, h_coord)]
                op_valid_tiles[h_coord] = {}
                op_valid_tiles[h_coord]['tiles'] = op_valids
                op_valid_tiles[h_coord]['head_value'] = snake.head_value
                op_valid_tiles[h_coord]['body_value'] = snake.body_value
            # print('op_valid_tiles')
            # pprint(op_valid_tiles)
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
            # for state in next_states:
                # print('next state')
                # self.print_map(state)
            return next_states

        def recurse(s_map, self_coord, body_coords, self_length, depth=0):
            if depth >= max_depth:
                return 0
            results = []
            if s_map[self_coord[1], self_coord[0]] == self.env.FOOD_TILE:
                self_length += 1
            for next_state_map in get_next_states(s_map.copy()):
                if valids_in_next := self.valid_tiles(next_state_map, self_coord):
                    sub_results = []
                    # self.print_map(next_state_map)
                    # print(self_coord)
                    # print(valids_in_next)
                    for self_valid in valids_in_next:
                        body_coords_copy = body_coords.copy()
                        old_tail = self.update_body(self_valid, body_coords_copy, self_length)
                        next_recurse_map = self.update_snake_position(next_state_map.copy(), body_coords_copy, old_tail)
                        area_check = self.area_check_wrapper(next_recurse_map, body_coords_copy, self_valid)
                        if not area_check['is_clear']:
                            results.append(1)
                            continue
                        # self.print_map(next_recurse_map)
                        # print('depth: ', depth)
                        # print(self.id)
                        result = recurse(next_recurse_map, self_valid, body_coords_copy, self_length, depth+1)
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
        self.print_map(next_state_map)
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
                        current_results=None,
                        rundata=None,
                        timeout_ms=None,
                        branch_common=None):
        safety_buffer = 3
        if branch_common is None:
            branch_common = {}
            branch_common['min_margin'] = 0
        if timeout_ms is None:
            timeout_ms = self.calc_timeout
        if start_time is None:
            start_time = time()
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

        area_checks = {}

        if valid_tiles:
            best_margin = -length
            target_tile = None
            for tile in valid_tiles:
                area_check = area_checks.get(tile, None)
                if area_check is None:
                    area_check = self.area_check_wrapper(s_map, body_coords, tile, target_margin=branch_common['min_margin'])
                    area_checks[tile] = area_check
                if area_check['margin'] > best_margin:
                    best_margin = area_check['margin']
                    best_results['margin'] = max(best_results['margin'], best_margin)
                    target_tile = tile
                # print('tile: ', tile, area_check)
            if target_tile is None:
                target_tile = self.target_tile(s_map, body_coords)
            valid_tiles.sort(key=lambda x: 0 if x == target_tile else 1)

            # print(" ")
            # self.print_map(s_map)
            # print({"food": [list(c) for c in self.env.food.locations], self.id: body_coords})
            # print(area_checks)
            # print("min_margin: ", branch_common['min_margin'])
            for tile in valid_tiles:
                if branch_common.get('min_margin', 0) > best_margin:
                    # print('margin break')
                    continue
                area_check = area_checks[tile].copy()
                # print(area_check)
                if area_check['has_tail']:
                    current_results['free_path'] = True
                    current_results['len_gain'] = area_check['food_count']
                    current_results['depth'] = length
                    current_results['margin'] = area_check['margin']
                    return current_results
                if not area_check['is_clear']:
                    best_results['depth'] = max(best_results['depth'], area_check['tile_count'])
                    best_results['margin'] = max(best_results['margin'], current_results['margin'])
                    # print('not clear')
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
                    branch_common=branch_common)
                if check_result['free_path'] or check_result['timeout']:
                    return check_result
                if len(valid_tiles) == 1:
                    branch_common['min_margin'] += 1
        return best_results
