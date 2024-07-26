import random
from statistics import mean
from time import time
import numpy as np

from snake_sim.snakes.utils import coord_op

from snakes.autoSnakeBase import AutoSnakeBase, copy_map
from snake_sim.snakes.snake_env import (
        DIR_MAPPING
    )


class AutoSnake2(AutoSnakeBase):
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


    def is_single_area(self, coord, areas):
        for area in areas:
            if coord in area and len(area) == 1:
                return True
        return False

    def get_area_info(self, s_map, body_coords, start_coord, checked=None):
        current_coords = [start_coord]
        safety_buffer = 1
        stats = {
            'area_start': start_coord,
            'food': 0,
            'tiles': 1,
            'might_escape': False,
            'needed_steps': 0,
            'total_steps': 1,
            'max_index': 0,
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
                    if n_coord in [x for area in unexplored_areas for x in area] or self.is_single_area(n_coord, areas):
                        for area in unexplored_areas:
                            if n_coord in area:
                                if not checked[n_y * self.env.width + n_x]:
                                    # print('doing subsearch')
                                    checked[n_y * self.env.width + n_x] = True
                                    stats['tiles'] += 1
                                    area_info = self.get_area_info(s_map, body_coords, n_coord, checked=checked)
                                    if area_info['might_escape']:
                                        stats['might_escape'] = True
                                        stats['needed_steps'] = area_info['needed_steps']
                                        stats['total_steps'] += area_info['total_steps']
                                        stats['food'] += area_info['food']
                                        stats['tiles'] += area_info['tiles']
                                        for k in area_info:
                                            if k not in stats:
                                                stats[k] = area_info[k]
                                    stats['max_index'] = max(area_info['max_index'], stats['max_index'])
                                    # print('stats: ', stats)
                                    # print('area_info:', area_info)
                                for a_x, a_y in area:
                                    checked[a_y * self.env.width + a_x] = True
                    else:
                        t_x, t_y = n_coord
                        if self.env.is_inside(n_coord):
                            # print('is_inside')
                            if not checked[t_y * self.env.width + t_x]:
                                # print('is_not_checked')
                                if s_map[t_y][t_x] in self.env.valid_tile_values:
                                    if s_map[c_y][c_x] == self.env.FOOD_TILE:
                                        stats['food'] += 1
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
            max_index = max(max(self_indexes), stats['max_index'])
            needed_steps = body_len - max_index + safety_buffer
            stats['max_index'] = max_index
            stats['self_indexes'] = self_indexes
            stats['body_len'] = body_len
            stats['needed_steps'] = needed_steps
            stats['total_steps'] = total_steps
            if total_steps >= needed_steps:
                stats['might_escape'] = True
                break

        # print('returning stats: ', stats)
        return stats




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
            # print(f"Getting area info for tile: {tile}")
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
        # if self.route is not None:
        #     route_copy = self.route + [target_tile]
        # self.show_route(self.map_to_print, route_copy)
        for tile in valid_tiles:
            # print("Checking option: ", tile)
            option_data = self.get_option_data(copy_map(self.map), tile)
            options[tile] = option_data
        # for option in options:
            # print(option, options[option])
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
            # print(f"best_option: {best_option}")
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

    #

