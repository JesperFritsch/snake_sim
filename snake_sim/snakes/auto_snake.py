import random
import numpy as np
import itertools
from typing import List
from collections import deque
from dataclasses import dataclass
from statistics import mean
from functools import cmp_to_key
from time import time
from pprint import pprint
import sys

from snake_sim.cpp_bindings.area_check import AreaChecker

from snake_sim.utils import coord_op, distance, exec_time, Coord

from snake_sim.snakes.auto_snake_base import AutoSnakeBase

class BFSFrame:
    def __init__(self,
                 try_coord,
                 s_map,
                 body_coords,
                 tiles_to_visit,
                 area_checks,
                 best_margin,
                 has_tail,
                 best_margin_over_edge,
                 has_safe_food_margin,
                 target_margin=0):
        """ target_margin is the minimum margin used when checking areas """
        self.try_coord = try_coord
        self.map = s_map
        self.body_coords = body_coords
        self.tiles_to_visit = tiles_to_visit
        self.area_checks = area_checks
        self.best_margin = best_margin
        self.target_margin = target_margin
        self.has_tail = has_tail
        self.visited_tiles = set()
        self.has_safe_food_margin = has_safe_food_margin
        self.best_margin_over_edge = best_margin_over_edge

    def get_next_tile(self):
        if self.tiles_to_visit:
            tile = self.tiles_to_visit.pop()
            self.visited_tiles.add(tile)
            return tile
        return None


class AutoSnake(AutoSnakeBase):
    TIME_LIMIT = True
    MAX_RISK_CALC_DEPTH = 3
    SAFE_MARGIN_FACTOR = 0.12

    def __init__(self, calc_timeout=1000):
        super().__init__()
        self.calc_timeout = calc_timeout
        self.food_in_route = []
        self.failed_paths = set()
        self.area_checker = None

    def _init_area_checker(self):
        self.area_checker = AreaChecker(
            self.env_data.food_value,
            self.env_data.free_value,
            self.body_value,
            self.head_value,
            self.env_data.width,
            self.env_data.height)

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


    def _explore_option(self, start_coord, target_margin=0, timeout_ms=None, exhaustive=False):
        # This is taking the risk of not finding that the snake cant use all the tiles in the area.
        safe_option = self._best_first_search(
                self.map.copy(),
                self.body_coords.copy(),
                start_coord,
                min_margin=target_margin,
                safe_margin_factor=self.SAFE_MARGIN_FACTOR,
                timeout_ms=timeout_ms,
                exhaustive=exhaustive)
        return safe_option


    def _get_best_option(self):
        """ return the best option to move to """
        options = {}
        best_option = None
        valid_tiles = self._valid_tiles(self.map, self.coord)
        area_checks = {}
        if not valid_tiles:
            return None
        area_checks = self._check_areas(self.map, self.body_coords, valid_tiles, exhaustive=True, safe_margin_factor=self.SAFE_MARGIN_FACTOR)
        valid_tiles.sort(key=lambda x: area_checks[x]['margin_over_edge'], reverse=True)
        for coord in valid_tiles:
            safe_option = self._explore_option(coord, exhaustive=True)
            options[coord] = safe_option
            if safe_option:
                break
        free_options = [coord for coord, o in options.items() if o]
        if free_options:
            best_option = max(free_options, key=lambda x: area_checks[x]['margin'])
        else:
            best_option = max(area_checks.items(), key=lambda x: x[1]['margin'])[0]
        if best_option is not None:
            return best_option
        else:
            return None

    def might_close_area(self, s_map, head_coord, neck_coord):
        """
        This should cover all scenarios where an area check is needed.
        checking if the head will close of any area.
        and checking if the head creates paths that are one tile wide.
        """
        free_value = self.env_data.free_value
        food_value = self.env_data.food_value
        head_dir = (head_coord[0] - neck_coord[0], head_coord[1] - neck_coord[1])
        perp_axis = (head_dir[1], head_dir[0])
        coord_ahead = (head_coord[0] + head_dir[0], head_coord[1] + head_dir[1])

        if not self.is_inside(coord_ahead):
            return True

        if s_map[coord_ahead[1], coord_ahead[0]] != free_value and s_map[coord_ahead[1], coord_ahead[0]] != food_value:
            return True
        besides_a = (head_coord[0] + perp_axis[0], head_coord[1] + perp_axis[1])
        besides_b = (head_coord[0] - perp_axis[0], head_coord[1] - perp_axis[1])
        diag_a_ahead = (coord_ahead[0] + perp_axis[0], coord_ahead[1] + perp_axis[1])
        diag_b_ahead = (coord_ahead[0] - perp_axis[0], coord_ahead[1] - perp_axis[1])

        if self.is_inside(diag_a_ahead) and self.is_inside(besides_a):
            if s_map[diag_a_ahead[1], diag_a_ahead[0]] != free_value and s_map[diag_a_ahead[1], diag_a_ahead[0]] != food_value:
                if s_map[besides_a[1], besides_a[0]] == free_value or s_map[besides_a[1], besides_a[0]] == food_value:
                    return True
            besides2 = (head_coord[0] + perp_axis[0] * 2, head_coord[1] + perp_axis[1] * 2)
            if s_map[besides_a[1], besides_a[0]] == free_value or s_map[besides_a[1], besides_a[0]] == food_value:
                if self.is_inside(besides2) and (s_map[besides2[1], besides2[0]] != free_value and s_map[besides2[1], besides2[0]] != food_value):
                    return True

        if self.is_inside(diag_b_ahead) and self.is_inside(besides_b):
            if s_map[diag_b_ahead[1], diag_b_ahead[0]] != free_value and s_map[diag_b_ahead[1], diag_b_ahead[0]] != food_value:
                if s_map[besides_b[1], besides_b[0]] == free_value or s_map[besides_b[1], besides_b[0]] == food_value:
                    return True
            besides2 = (head_coord[0] - perp_axis[0] * 2, head_coord[1] - perp_axis[1] * 2)
            if s_map[besides_b[1], besides_b[0]] == free_value or s_map[besides_b[1], besides_b[0]] == food_value:
                if self.is_inside(besides2) and (s_map[besides2[1], besides2[0]] != free_value and s_map[besides2[1], besides2[0]] != food_value):
                    return True
        return False

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

    def get_locations(self, s_map, value):
        coordinate = np.where(s_map == value)
        return [(int(col), int(row)) for row, col in zip(coordinate[0], coordinate[1])]


    def _get_closest_accessible_food_route(self):
        s_map = self.map.copy()
        food_locations = self.get_locations(s_map, self.env_data.food_value)
        route = self._get_route(self.map, self.coord, target_tiles=[l for l in food_locations if l != self.coord])
        while route:
            route = self._fix_route(route)
            # return route
            if self._check_safe_food_route(s_map, route):
                break
            else:
                food_locations.remove(route[0])
            route = self._get_route(self.map, self.coord, target_tiles=[l for l in food_locations if l != self.coord])
        return route


    def _get_future_available_food_map(self):
        s_map = self.map.copy()
        valid_tiles = self._valid_tiles(self.map, self.coord)
        future_valids = {coord: self._valid_tiles(self.map, coord) for coord in valid_tiles}
        food_map = {}
        all_area_checks = {}
        all_clear_checks = {}
        additonal_food = {}
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
            additonal_food[coord] = old_map_value == self.env_data.food_value
        all_checks = [a for check in all_area_checks.values() for a in check]
        combine_food = all([a['margin'] >= a['food_count'] and a["food_count"] > 0 for a in all_checks])
        # combine_food = False
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
        areas_map = {coord: self._area_check_wrapper(s_map, self.body_coords, coord, safe_margin_factor=self.SAFE_MARGIN_FACTOR) for coord in valid_tiles}
        return areas_map


    def _pick_direction(self):
        if self.area_checker is None:
            self._init_area_checker()
        next_tile = None
        planned_tile = None
        planned_route = None
        closest_food_route = self._get_closest_accessible_food_route()
        # print("closest_food_route: ", closest_food_route)
        if closest_food_route:
            planned_route = closest_food_route
            planned_tile = planned_route.pop()
        valid_tiles = self._valid_tiles(self.map, self.coord)
        if planned_tile: # and self.might_close_area(self.map, planned_tile, self.coord):
            food_map = self._get_future_available_food_map()
        else:
            food_map = {k: 0 for k in valid_tiles}
        # print("food_map: ", food_map)
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
            planned_area = self._area_check_wrapper(self.map, self.body_coords, planned_tile, safe_margin_factor=self.SAFE_MARGIN_FACTOR)
            # print("planned_tile: ", planned_tile)
            # print("planned_area: ", planned_area)
            if planned_area['margin'] > planned_area['food_count'] and planned_area['margin_over_edge'] > self.SAFE_MARGIN_FACTOR:
                # print("explore_option")
                safe_option = self._explore_option(planned_tile)
                # print("safe_option: ", safe_option)
                if safe_option:
                    next_tile = planned_tile
        if next_tile is None:
            # print("get_best_option")
            option = self._get_best_option()
            if option:
                next_tile = option
        return next_tile

    def _target_tile(self, body_coords):
        self_coord = body_coords[0]
        s_dir = coord_op(self_coord, body_coords[1], '-')
        return coord_op(self_coord, s_dir, '+')

    def _area_check_wrapper(self, s_map, body_coords, start_coord, target_margin=0, food_check=False, exhaustive=False, safe_margin_factor=0.0):
        return self.area_checker.area_check(s_map, list(body_coords), start_coord, target_margin, food_check, exhaustive, float(safe_margin_factor))

    def _create_bfs_frame(self, new_coord, s_map, body_coords, min_margin=0, safe_margin_factor=0, safe_food_margin=0, exhaustive=False):
        body_copy = body_coords.copy()
        map_copy = s_map.copy()
        length = len(body_coords)
        if s_map[new_coord[1], new_coord[0]] == self.env_data.food_value:
            length += 1
        old_tail = self.update_body(new_coord, body_copy, length)
        self._update_snake_position(map_copy, body_copy, old_tail)
        valid_tiles = self._valid_tiles(map_copy, new_coord)
        best_margin = -len(body_coords)
        best_margin_over_edge = 0
        has_safe_food_margin = True
        target_margin = max(min_margin, safe_food_margin)
        area_checks = self._check_areas(map_copy, body_copy, valid_tiles, target_margin=target_margin, safe_margin_factor=safe_margin_factor, exhaustive=exhaustive)
        if area_checks:
            best_margin = max([a['margin'] for a in area_checks.values()])
            best_margin_over_edge = max([a['margin_over_edge'] for a in area_checks.values()])
            tiles_to_visit = [t for t in valid_tiles if area_checks[t]['is_clear']]
            tiles_to_visit.sort(key=lambda x: area_checks[x]['margin'])
            has_safe_food_margin = any([a['margin'] >= safe_food_margin for a in area_checks.values()])
            has_tail = any([a['has_tail'] for a in area_checks.values()])
        else:
            tiles_to_visit = []
            has_tail = False

        return BFSFrame(
            new_coord,
            map_copy,
            body_copy,
            tiles_to_visit,
            area_checks,
            best_margin=best_margin,
            target_margin=min_margin,
            has_tail=has_tail,
            best_margin_over_edge=best_margin_over_edge,
            has_safe_food_margin=has_safe_food_margin
        )

    def _check_areas(self, s_map, body_coords, tiles, target_margin=0, **kwargs):
        areas = {}
        for tile in tiles:
            area_check = self._area_check_wrapper(s_map, body_coords, tile, target_margin=target_margin, **kwargs)
            areas[tile] = area_check
        return areas

    def _best_first_search(self,
                           s_map,
                           body_coords,
                           new_coord,
                           min_margin=0,
                           safe_margin_factor=0,
                           min_depth=2,
                           timeout_ms=None,
                           rundata=None,
                           exhaustive=False):
        if timeout_ms is None:
            timeout_ms = self.calc_timeout
        search_stack: List[BFSFrame] = []

        first_bfs_frame = self._create_bfs_frame(
            new_coord,
            s_map,
            body_coords,
            min_margin,
            safe_margin_factor=safe_margin_factor,
            exhaustive=exhaustive)

        if first_bfs_frame.area_checks:
            max_food = max([a['food_count'] for a in first_bfs_frame.area_checks.values()])
        else:
            max_food = 0
        search_stack.append(first_bfs_frame)
        start_time = time()
        while search_stack:
            frame: BFSFrame = search_stack[-1]
            s_map = frame.map
            new_coord = frame.try_coord
            body_coords = frame.body_coords

            if rundata is not None:
                rundata.append(body_coords.copy())

            if ((time() - start_time) * 1000 > timeout_ms) and self.TIME_LIMIT:
                return False

            # if the margin is large enough compared to the number of tiles left, then we can assume that we will fit in the area
            # and we dont need to actually find a path to the end
            # but that only works if the number of tiles left is large enough, if an area has 5 tiles and the margin is 1
            # then best_margin_over_edge will be 0.2 which might be over the limit, but there is a high risk of unreachables
            # self.print_map(s_map)
            # for a_coord, area in frame.area_checks.items():
            #     print("area: ", a_coord, " margin: ", area['margin'], " margin_over_edge: ", area['margin_over_edge'], " food_count: ", area['food_count'], " is_clear: ", area['is_clear'], " has_tail: ", area['has_tail'])
            # print("frame: ", frame.try_coord, " margin: ", frame.best_margin, " margin_over_edge: ", frame.best_margin_over_edge, " has_tail: ", frame.has_tail, " has_safe_food_margin: ", frame.has_safe_food_margin)
            if frame.has_safe_food_margin:
                if (frame.best_margin_over_edge >= safe_margin_factor and
                    frame.best_margin > frame.best_margin_over_edge * 40) and len(search_stack) >= min_depth:
                    return True

                if len(search_stack) > len(body_coords) + 1 or frame.has_tail:
                    return True
                next_tile = frame.get_next_tile()
            else:
                if exhaustive:
                    next_tile = None
                else:
                    return False

            # print("next_tile: ", next_tile)
            if next_tile is None:
                if (len(frame.visited_tiles) + len(frame.tiles_to_visit)) == 1:
                    min_margin += 1
                    for s_frame in reversed(search_stack):
                        if s_frame.best_margin < min_margin or not s_frame.tiles_to_visit:
                            search_stack.pop()
                        else:
                            break
                else:
                    search_stack.pop()
            else:
                next_frame = self._create_bfs_frame(
                    next_tile,
                    s_map,
                    body_coords,
                    min_margin,
                    safe_margin_factor=safe_margin_factor,
                    safe_food_margin=max_food,
                    exhaustive=False)
                # print("next_frame: ", next_frame.try_coord)
                search_stack.append(next_frame)
            # print("#####################")
            # print("")

        return False

