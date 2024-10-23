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

from .auto_snake_base import AutoSnakeBase

class AutoSnake(AutoSnakeBase):
    TIME_LIMIT = True
    MAX_RISK_CALC_DEPTH = 3

    def __init__(self, id: str, start_length: int, calc_timeout=1000):
        super().__init__(id, start_length)
        self.calc_timeout = calc_timeout
        self.food_in_route = []
        self.failed_paths = set()
        self.area_checker = None


    def _init_after_bind(self):
        self.area_checker = AreaChecker(
            self.env.FOOD_TILE,
            self.env.FREE_TILE,
            self.body_value,
            self.env.width,
            self.env.height)


    def _fix_route(self, route, s_coord=None, valid_tiles=None):
        valid_tiles = valid_tiles or self._valid_tiles(self.map, self.coord)
        s_coord = s_coord or self.coord
        # print("Coord: ", s_coord)
        if s_coord in route and len(route) > 1:
            route = deque(list(route)[:route.index(s_coord)])
        try:
            if route[-1] not in valid_tiles:
                try:
                    sub_route = self._get_route(self.map, s_coord, end=route[-1])
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


    def _explore_option(self, start_coord, food_ahead=None, timeout_ms=None):
        time_s = time()
        branch_common = None
        if food_ahead is not None:
            branch_common = {}
            branch_common['min_margin'] = food_ahead
        option = self._deep_look_ahead(
                self.map.copy(),
                start_coord,
                self.body_coords.copy(),
                self.length,
                start_time=time_s,
                timeout_ms=timeout_ms,
                branch_common=branch_common)
        option['coord'] = start_coord
        option['risk'] = 0
        return option


    def _get_best_option(self):
        """ return the best option to move to """
        options = {}
        best_option = None
        valid_tiles = self._valid_tiles(self.map, self.coord)
        area_checks = {}
        target_tile = None
        best_margin = -self.length
        if not valid_tiles:
            return None
        for tile in valid_tiles:
            area_checks[tile] = self._area_check_wrapper(self.map, self.body_coords, tile, exhaustive=True)
            if area_checks[tile]['margin'] > best_margin:
                best_margin = area_checks[tile]['margin']
                target_tile = tile
        valid_tiles.sort(key=lambda x: 0 if x == target_tile else 1)
        for coord in valid_tiles:
            option = self._explore_option(coord)
            options[coord] = option
            if option['free_path'] and area_checks[coord]['margin'] >= area_checks[coord]['food_count']:
                break
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


    def _check_safe_food_route(self, s_map, food_route):
        end_coord = food_route[0]
        body_copy = self.body_coords.copy()
        map_copy = s_map.copy()
        for coord in reversed(food_route):
            old_tail = self.update_body(coord, body_copy, self.length + 1)
            self._update_snake_position(map_copy, body_copy, old_tail)
        valid_tiles = self._valid_tiles(map_copy, end_coord)
        area_checks = [self._area_check_wrapper(map_copy, body_copy, tile) for tile in valid_tiles]
        return any([a["margin"] >= a["food_count"] for a in area_checks])


    def _get_closest_accessible_food_route(self):
        s_map = self.map.copy()
        food_locations = self.env.food.locations.copy()
        route = None
        while route := self._get_route(self.map, self.coord, target_tiles=[l for l in food_locations if l != self.coord]):
            route = self._fix_route(route)
            if self._check_safe_food_route(s_map, route):
                return route
            else:
                food_locations.remove(route[0])
        return route


    def _get_future_available_food_map(self):
        s_map = self.map.copy()
        valid_tiles = self._valid_tiles(self.map, self.coord)
        future_valids = {coord: self._valid_tiles(self.map, coord) for coord in valid_tiles}
        food_map = {}
        all_area_checks = {}
        all_clear_checks = {}
        additonal_food = {}
        best_checks = []
        for coord, valids in future_valids.items():
            if not valids:
                continue
            x, y = coord
            body_coords_copy = self.body_coords.copy()
            map_copy = s_map.copy()
            old_map_value = map_copy[y, x]
            old_tail = self.update_body(coord, body_coords_copy, self.length)
            self._update_snake_position(map_copy, body_coords_copy, old_tail)
            area_checks = [self._area_check_wrapper(map_copy, body_coords_copy, tile, food_check=True) for tile in valids]
            clear_checks = [a for a in area_checks if a['is_clear']]
            all_area_checks[coord] = area_checks
            if clear_checks:
                all_clear_checks[coord] = clear_checks
            additonal_food[coord] = old_map_value == self.env.FOOD_TILE
        all_checks = [a for check in all_area_checks.values() for a in check]
        combine_food = all([a['margin'] >= a['food_count'] and a["food_count"] > 0 for a in all_checks])
        combine_food = False
        if all_checks:
            combined_food = max([a['food_count'] for a in all_checks])
        else:
            combined_food = 0
        for coord, area_checks in all_clear_checks.items():
            if combine_food:
                most_food = combined_food
            else:
                most_food = max(area_checks, key=lambda x: x['food_count'])['food_count']
            food_map[coord] = most_food + (1 if additonal_food[coord] else 0)
        return food_map


    def _get_available_areas(self):
        s_map = self.map.copy()
        valid_tiles = self._valid_tiles(self.map, self.coord)
        areas_map = {coord: self._area_check_wrapper(s_map, self.body_coords, coord) for coord in valid_tiles}
        return areas_map


    def _pick_direction(self):
        next_tile = None
        planned_tile = None
        planned_route = None
        closest_food_route = self._get_closest_accessible_food_route()
        if closest_food_route:
            planned_route = closest_food_route
            planned_tile = planned_route.pop()
        areas_map = self._get_available_areas()
        food_map = self._get_future_available_food_map()
        if len(self.env.alive_snakes) > 1:
            max_food = max([x for x in food_map.values()] or [0])
            food_map = {k: max_food for k in food_map}
        if food_map and planned_tile is not None:
            best_food_pair = max(food_map.items(), key=lambda x: x[1])
            max_food_value = max(food_map.values())
            best_food_pairs = [pair for pair in food_map.items() if pair[1] == max_food_value]
            def cmp(c1, c2):
                return distance(planned_tile, c1[0]) - distance(planned_tile, c2[0])
            best_food_pair = min(best_food_pairs, key=cmp_to_key(cmp))
            best_food_tile, best_food_value = best_food_pair
            planned_tile_food_value = food_map.get(planned_tile, 0)
            if planned_tile is None or planned_tile_food_value < best_food_value:
                planned_tile = best_food_tile
            planned_area = areas_map[planned_tile]
            if planned_area['margin'] >= planned_area['food_count']:
                option = self._explore_option(planned_tile, food_ahead=planned_area['food_count'])
                if option['free_path']:
                    next_tile = planned_tile
        if next_tile is None:
            option = self._get_best_option()
            if option:
                next_tile = option['coord']
        return next_tile


    def _target_tile(self, body_coords):
        self_coord = body_coords[0]
        s_dir = coord_op(self_coord, body_coords[1], '-')
        return coord_op(self_coord, s_dir, '+')


    def _area_check_wrapper(self, s_map, body_coords, start_coord, target_margin=0, food_check=False, exhaustive=False):
        return self.area_checker.area_check(s_map, list(body_coords), start_coord, target_margin, food_check, exhaustive)


    def _deep_look_ahead(self, s_map, new_coord, body_coords, length,
                        start_time=None,
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
        s_map = self._update_snake_position(s_map, body_coords, old_tail)
        valid_tiles = self._valid_tiles(s_map, new_coord)

        #rundata is just for when i want to generate frames of how the algorithm searches in test/tests/autosnake2_test.py
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
                    area_check = self._area_check_wrapper(s_map, body_coords, tile, target_margin=branch_common['min_margin'])
                    area_checks[tile] = area_check
                if area_check['margin'] > best_margin:
                    best_margin = area_check['margin']
                    best_results['margin'] = max(best_results['margin'], best_margin)
                    target_tile = tile
            if target_tile is None:
                target_tile = self._target_tile(body_coords)
            valid_tiles.sort(key=lambda x: 0 if x == target_tile else 1)

            for tile in valid_tiles:
                if branch_common.get('min_margin', 0) > best_margin:
                    continue
                area_check = area_checks[tile].copy()

                if area_check['has_tail']:
                    current_results['free_path'] = True
                    current_results['len_gain'] = area_check['food_count']
                    current_results['depth'] = length
                    current_results['margin'] = area_check['margin']
                    return current_results

                if not area_check['is_clear']:
                    best_results['depth'] = max(best_results['depth'], area_check['tile_count'])
                    best_results['margin'] = max(best_results['margin'], current_results['margin'])
                    continue
                current_results['margin'] = area_check['margin']

                check_result = self._deep_look_ahead(
                    s_map.copy(),
                    tile,
                    body_coords.copy(),
                    length,
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
