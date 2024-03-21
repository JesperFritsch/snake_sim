import random
import array
import math
import itertools
from time import time
from snakes.snake import Snake
from statistics import mean
from snake_env import (
        coord_op,
        DIR_MAPPING
    )


def copy_map(s_map):
    return [[_ for _ in row] for row in s_map]

class AutoSnake(Snake):
    TIME_LIMIT = True
    MAX_RECURSE_DEPTH = 100
    MAX_RISK_CALC_DEPTH = 3
    MAX_BRANCH_TIME = 100
    MAX_UPDATE_TIME = 15000
    MAX_SEARCH_TIME = 200

    def __init__(self, id: str, start_length: int):
        super().__init__(id, start_length)
        self.time_step = 0
        self.height = None
        self.width = None
        self.env
        self.x = None
        self.y = None
        self.route = None
        self.start_time = 0
        self.map_counters = None
        self.snake_lengths = None
        self.map_to_print = None
        self.length = start_length
        self.alive_opps = []

    def in_sight(self, head_coord, coord, sight_len=2):
        h_x, h_y = head_coord
        x, y = coord
        return (h_x - sight_len) <= x <= (h_x + sight_len) and (h_y - sight_len) <= y <= (h_y + sight_len)

    def update(self):
        self.start_time = time()
        self.time_step += 1
        if self.map_counters is None:
            self.map_counters = [2] * (self.height * self.width)
        self.update_map(self.env.map)
        self.map_to_print = copy_map(self.map)
        self.update_survivors()
        tile = self.pick_direction()
        # self.print_map(self.map_to_print)
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
            if self.x < self.width - 1 and s_map[self.y + 1][self.x + 1] in self.alive_opps:
                moves.append((1, 0))
            elif self.x > 0 and s_map[self.y + 1][self.x - 1] in self.alive_opps:
                moves.append((-1, 0))
        elif head_dir == 'right':
            if self.y > 0 and s_map[self.y - 1][self.x - 1] in self.alive_opps:
                moves.append((0, -1))
            elif self.y < self.height - 1 and s_map[self.y + 1][self.x - 1] in self.alive_opps:
                moves.append((0, 1))
        elif head_dir == 'down':
            if self.x < self.width - 1 and s_map[self.y - 1][self.x + 1] in self.alive_opps:
                moves.append((1, 0))
            elif self.x > 0 and s_map[self.y - 1][self.x - 1] in self.alive_opps:
                moves.append((-1, 0))
        else:
            if self.y > 0 and s_map[self.y - 1][self.x + 1] in self.alive_opps:
                moves.append((0, -1))
            elif self.y < self.height - 1 and s_map[self.y + 1][self.x + 1] in self.alive_opps:
                moves.append((0, 1))
        return tuple(moves)

    def show_route(self, s_map, s_route):
        if s_route is None: return
        dest = s_route[0]
        for x, y in s_route[1:]:
            s_map[y][x] = '¤'
        return copy_map(s_map)

    def update_snake_position(self, s_map, body_coords, old_tail):
        head = body_coords[0]
        if old_tail is not None:
            s_map[old_tail[1]][old_tail[0]] = self.env.FREE_TILE
        for i in range(2):
            x, y = body_coords[i]
            s_map[y][x] = self.head_value if (x, y) == head else self.body_value
        return copy_map(s_map)

    def update_map(self, flat_map):
        if self.map is None:
            self.map = [array.array('B', [self.env.FREE_TILE] * self.env.width) for _ in range(self.env.height)]
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

    def get_area_info(self, s_map, body_coords, start_coord=None, current_coords=None, checked=None, stats=None):
        if current_coords is None:
            if start_coord is not None:
                current_coords = [start_coord]
            else:
                ValueError('No start coord provided')
        if stats is None:
            stats = {
                'food': 0,
                'tiles': 0,
                'has_tail': False
            }
        if checked is None:
            checked = [False] * (self.height * self.width)
        next_coords = []
        for coord in current_coords:
            x, y = coord
            neighbour_coords = self.neighbours(coord)
            if s_map[y][x] == self.env.FOOD_TILE:
                stats['food'] += 1
            for coord in neighbour_coords:
                if self.is_valid_tile(s_map, coord):
                    t_x, t_y = coord
                    if not checked[t_y * self.width + t_x]:
                        next_coords.append(coord)
                        checked[t_y * self.width + t_x] = True
                elif coord == body_coords[-1]:
                    stats['has_tail'] = True
        if next_coords:
            stats['tiles'] += len(next_coords)
            stats = self.get_area_info(s_map, body_coords=body_coords, current_coords=next_coords, checked=checked, stats=stats)
        return stats

    def closest_apple_route(self, current_coords, s_map, checked=None, depth=0, head_coord=None):
        if checked is None:
            checked = [False] * (self.height * self.width)
        if len(current_coords) == 1 and head_coord is None:
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
                if not checked[t_y * self.width + t_x]:
                    next_coords.append(valid_coord)
                    coord_map[valid_coord] = coord
                    checked[t_y * self.width + t_x] = True
        if next_coords:
            sub_route = self.closest_apple_route(next_coords, s_map, checked=checked, depth=depth+1)
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

    def get_option_data(self, s_map, body_coords, head_coord, option_coord):
        t_dir = coord_op(option_coord, head_coord, '-')
        time_s = time()
        option = self.recurse_check_option(copy_map(s_map), option_coord, body_coords.copy(), self.length, start_time=time_s)
        # print(f'recurse_time for {option_coord}: ', (time() - time_s) * 1000)
        option['coord'] = option_coord
        option['free_path'] = option['depth'] >= self.length
        option['dir'] = t_dir
        # time_s = time()

        time_s = time()
        option['risk'] = self.calc_immediate_risk(copy_map(s_map), option_coord)
        # print(f'calc_risk for {option_coord}: ', (time() - time_s) * 1000)

        # option['risk'] = 0
        return option

    def pick_direction(self):
        valid_tiles = self.valid_tiles(self.map, self.coord)
        # print('areas: ', self.areas(self.map, self.coord, valid_tiles))
        random.shuffle(valid_tiles)
        options = {}
        time_s = time()
        target_tile = self.target_tile(self.map, self.coord)
        # print(f"Time to target:", (time() - time_s) * 1000)
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
                good_options = risk_free_options
            else:
                if free_options:
                    best_option = min(free_options, key=lambda x: x['risk'])
                else:
                    best_option = max(options.values(), key=lambda x: x['len_gain'])
                # print('best_option: ', best_option)
                if best_option is not None:
                    return best_option['coord']
                else:
                    return None
            if free_good_options := [o for o in good_options if o['free_path']]:
                #find optinos with the best length gain
                best_len_gain = max(o['len_gain'] for o in free_good_options)
                best_len_opts = [o for o in free_good_options if o['len_gain'] == best_len_gain]
                best_early_gain = min(sum(o['apple_time']) for o in best_len_opts)
                #out of those options find the one that has the gain earlier
                best_early_gain_opts = [o for o in best_len_opts if sum(o['apple_time']) == best_early_gain]
                best_option = best_early_gain_opts[0]
                if target_option and target_option['free_path']:
                    if target_option['len_gain'] == best_len_gain \
                    and sum(target_option['apple_time']) == best_early_gain \
                    and target_option['risk'] == 0:
                        best_option = target_option
            elif free_options:
                best_option = min(free_options, key=lambda x: x['risk'])
            elif options:
                if (best_gain_option := max(options.values(), key=lambda x: x['len_gain']))['len_gain'] != 0:
                    best_option = best_gain_option
                else:
                    best_option = max(options.values(), key=lambda x: x['depth'])
        # print('best_option: ', best_option)
        if best_option is not None:
            return best_option['coord']
        return None

    def target_tile(self, s_map, head_coord, recurse_mode=False):
        if recurse_mode:
            if self.route:
                # print('Follow route')
                target_tile = self.route.pop()
                if not self.route: self.route = None
                return target_tile
        if self.env.FOOD_TILE in [x for row in s_map for x in row]:
            # print('Getting new route')
            if s_route := self.closest_apple_route([head_coord], s_map):
                self.route = s_route
                tile = tuple(self.route.pop())
                if not self.route: self.route = None
                return tile
        if not recurse_mode and (attack_moves := self.find_attack_moves(s_map)):
            # print('Attack!')
            return coord_op(self.coord, attack_moves[0], '+')
        # print('Same direction')
        s_dir = coord_op(self.coord, self.body_coords[1], '-')
        return coord_op(self.coord, s_dir, '+')
        # return random.choice(list(DIR_MAPPING.keys()))

    def print_map_counters(self, counters):
        for y in range(self.height):
            for x in range(self.width):
                print(f'{counters[y * self.width + x]: ^3}', end='')
            print()

    def print_map(self, s_map):
        for row in s_map:
            print(''.join([f' {chr(c)} ' for c in row]))

    def areas(self, s_map, s_coord, valid_tiles):
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
            if s_map[c_y][c_x] in self.env.valid_tile_values:
                areas[a] = areas[a] + [coord2]
            else:
                a += 1
                areas[a] = areas.get(a, []) + [coord2]
        return [tuple(area) for area in areas.values()]

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

    def is_valid_tile(self, s_map, coord):
        x, y = coord
        if self.env.is_inside(coord):
            return s_map[y][x] in self.env.valid_tile_values
        else:
            return False

    def neighbours(self, coord):
        return [coord_op(coord, s_dir, '+') for s_dir in DIR_MAPPING]

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


    def recurse_check_option(self, s_map, new_coord, body_coords, length, start_time, depth=1, best_results=None, current_results=None, area_checked=False):
        if current_results is None:
            current_results = {}
        if best_results is None:
            best_results = {}
        current_results['apple_time'] = current_results.get('apple_time', [])
        if s_map[new_coord[1]][new_coord[0]] == self.env.FOOD_TILE:
            length += 1
            current_results['apple_time'] = current_results['apple_time'] + [depth]

        current_results['depth'] = depth
        current_results['len_gain'] = length - self.length
        old_tail = self.update_body(new_coord, body_coords, length)
        s_map = self.update_snake_position(s_map, body_coords, old_tail)
        target_tile = self.target_tile(s_map, new_coord, recurse_mode=True)
        valid_tiles = self.valid_tiles(s_map, new_coord)
        best_results['depth'] = max(best_results.get('depth', 0), current_results['depth'])
        best_results['len_gain'] = max(best_results.get('len_gain', 0), current_results['len_gain'])
        best_results['apple_time'] = min(best_results, current_results, key=lambda x: x.get('apple_time', [length]))['apple_time']
        # best_results['apple_time'] = min(sum(best_results.get('apple_time', [length])), sum(current_results['apple_time']))
        valid_tiles.sort(key=lambda x: 0 if x == target_tile else 1)
        if ((time() - start_time) * 1000 > self.MAX_BRANCH_TIME) and self.TIME_LIMIT:
            return current_results
        if current_results.get('depth', 0) >= length:
            return current_results
        # print('______________________')
        # print('new_coord: ', new_coord)
        # print('target_tile:', target_tile)
        # print('valid_tiles: ', valid_tiles)
        # print('recurse_map')
        # print(self.route)
        # self.print_map(s_map)
        # print('recurse_time: ', (time() - self.start_time) * 1000)
        # print('______________________')
        # s_time = time()
        # print('areas_info:', areas_info)
        # print('area_info time: ', (time() - s_time) * 1000)
        if valid_tiles:
            for tile in valid_tiles:
                check_result = self.recurse_check_option(
                    copy_map(s_map),
                    tile,
                    body_coords.copy(),
                    length,
                    depth=depth+1,
                    best_results=best_results,
                    current_results=current_results.copy(),
                    area_checked=area_checked,
                    start_time=start_time)
                self.route = None
                if check_result.get('depth', 0) >= length:
                    return check_result
        return best_results