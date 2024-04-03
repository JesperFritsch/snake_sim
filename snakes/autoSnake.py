import random
import numpy as np
import itertools
from statistics import mean
from time import time
from utils import coord_op

from snakes.autoSnakeBase import AutoSnakeBase, copy_map
from snake_env import (
        DIR_MAPPING
    )

class AutoSnake(AutoSnakeBase):
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


    def get_area_info(self, s_map, body_coords, start_coord):
        current_coords = [start_coord]
        stats = {
            'food': 0,
            'tiles': 1,
            'might_escape': False
        }
        checked = np.array([False] * (self.env.height * self.env.width), dtype=bool)
        self_indexes = [0]
        body_len = len(body_coords)
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
                                    stats['tiles'] += 1
                                    next_coords.append(coord)
                            elif s_map[t_y][t_x] == self.body_value:
                                self_indexes.append(body_coords.index(coord))
                current_coords = next_coords
            total_length = stats['tiles'] - stats['food']
            max_index = max(self_indexes)
            needed_length = body_len - max_index
            if total_length >= needed_length:
                stats['might_escape'] = True
                break
        if len(self_indexes) == 1:
            #If the snake is not in the area, it might escape if the area is larger than the snake
            #Why 10? I don't know, it just works
            stats['might_escape'] = total_length >= body_len + 10
        return stats



    def get_option_data(self, s_map, body_coords, head_coord, option_coord):
        t_dir = coord_op(option_coord, head_coord, '-')
        time_s = time()
        option = self.recurse_check_option(copy_map(s_map), option_coord, body_coords.copy(), self.length, start_time=time_s, route=self.route)
        option['coord'] = option_coord
        option['timeout'] = option.get('timeout', False)
        option['free_path'] = option['depth'] >= self.length
        option['dir'] = t_dir

        time_s = time()
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
        for coord in valid_tiles:
            option = self.get_option_data(self.map, self.body_coords, self.coord, coord)
            options[coord] = option
        target_option = options.get(target_tile, None)
        free_options = [o for o in options.values() if o['free_path']]
        # print('self: ', self.coord)
        # print(f"{target_option=}")
        # print(f"{valid_tiles=}")
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
                best_early_gain = min(sum(o['apple_time'][:best_len_gain]) for o in best_len_opts)
                best_early_gain_opts = [o for o in best_len_opts if sum(o['apple_time']) >= best_early_gain]
                best_option = best_early_gain_opts[0]
                if target_option in best_early_gain_opts:
                    best_option = target_option
            else:
                if free_options:
                    best_option = min(free_options, key=lambda x: x['risk'])
                else:
                    best_len_gain = max(o['len_gain'] for o in options.values())
                    if best_len_gain == 0:
                        best_option = max(options.values(), key=lambda x: x['depth'])
                    else:
                        best_option = max(options.values(), key=lambda x: x['len_gain'])
        # print('best_option: ', best_option)
        if best_option is not None:
            return best_option['coord']
        return None

    #
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
            for tile in valid_tiles:
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