import random
import array
import copy
import json
from snakes.snake import Snake
from time import time
from dataclasses import dataclass, field
from collections import deque

UPDATE_TIME_MS = 250

DIR_MAPPING = {
    (0, -1): 'up',
    (1,  0): 'right',
    (0,  1): 'down',
    (-1, 0): 'left'
}

def coord_op(coord_left, coord_right, op):
        # Check the operation and perform it directly
        if op == '+':
            return tuple(l + r for l, r in zip(coord_left, coord_right))
        elif op == '-':
            return tuple(l - r for l, r in zip(coord_left, coord_right))
        elif op == '*':
            return tuple(l * r for l, r in zip(coord_left, coord_right))
        else:
            raise ValueError("Unsupported operation")

def right_shift_in(list_x, element):
    list_x.pop()
    list_x.insert(0, element)
    return list_x

def copy_map(s_map):
    return [[_ for _ in row] for row in s_map]

def copy_coords(coords):
    return [tuple([_ for _ in coord]) for coord in coords]


@dataclass
class Food:

    width: int
    height: int
    max_food: int
    locations: set = field(default_factory=set)

    def generate_new(self, s_map) -> list:
        empty_tiles = []
        for i in range(self.width * self.height):
            if s_map[i] == SnakeEnv.FREE_TILE:
                x, y = i % self.width, i // self.width
                empty_tiles.append((x, y))
        for _ in range(len(self.locations), self.max_food):
            new_food = random.choice(empty_tiles)
            empty_tiles.remove(new_food)
            self.add_new(new_food)
        for location in self.locations:
            x, y = location
            s_map[y * self.width + x] = SnakeEnv.FOOD_TILE


    def add_new(self, coord):
        self.locations.add(coord)

    def remove_eaten(self, coord):
        if coord in self.locations:
            self.locations.remove(coord)


class SnakeEnv:
    valid_tile_values = (FOOD_TILE, FREE_TILE) = (70, 46)
    COLOR_MAPPING = {
        FOOD_TILE: (223, 163, 49),
        FREE_TILE: (0, 0, 0)
    }

    def __init__(self, width, height, food) -> None:
        self.map = None
        self.snakes = {}
        self.width = width
        self.height = height
        self.time_step = 0
        self.alive_snakes = []
        self.map = self.fresh_map()
        self.snakes_info = {}
        self.food = Food(width=width, height=height, max_food=food)

    def fresh_map(self):
        return array.array('B', [self.FREE_TILE] * (self.width * self.height))

    def print_map(self):
        print(f"{'':@<{self.width*3}}")
        for y in range(self.height):
            print(''.join([f' {chr(c)} ' for c in self.map[y*self.width:y*self.width+self.width]]))

    def add_snake(self, snake, h_color, b_color):
        print(h_color, b_color)
        if isinstance(snake, Snake):
            snake.bind_env(self)
            rand_x = round(random.random() * (self.width - 1)) - 1
            rand_y = round(random.random() * (self.height - 1)) - 1
            snake.set_init_coord((rand_x, rand_y))
            head_value = snake.head_value
            body_value = snake.body_value
            self.COLOR_MAPPING[head_value] = h_color
            self.COLOR_MAPPING[body_value] = b_color
            if self.snakes.get(snake.id.upper(), None) is None:
                self.alive_snakes.append(snake)
                self.snakes_info[snake.id] = {
                    'length': snake.length,
                    'current_coord': snake.coord,
                    'last_coord': snake.coord,
                    'alive': True,
                    'id': snake.id.upper(),
                    'head_value': head_value,
                    'body_value': body_value,
                    'h_color': h_color,
                    'b_color': b_color,
                    'last_food': None
                }
                self.snakes[snake.id] = snake
            else:
                raise ValueError(f"Obj: {repr(snake)} has the same id as Obj: {repr(self.snakes[snake.id])}")
        else:
            raise ValueError(f"Obj: {repr(snake)} is not of type {Snake}")

    def remove_snake(self, snake_id):
        del self.snakes[snake_id]

    def is_inside(self, coord):
        x, y = coord
        return (0 <= x < self.width and 0 <= y < self.height)

    def update_snake_on_map(self, snake):
        head = snake.body_coords[0]
        old_tail = self.snakes_info[snake.id].get('old_tail', None)
        if old_tail is not None and old_tail != head:
            o_x, o_y = old_tail
            self.map[o_y * self.width + o_x] = self.FREE_TILE
        for i in range(2):
            x, y = snake.body_coords[i]
            self.map[y * self.width + x] = snake.head_value if (x, y) == head else snake.body_value

    def put_snake_on_map(self, snake):
        for coord in snake.body_coords:
            x, y = coord
            self.map[y * self.width + x] = snake.body_value
        x, y = snake.coord
        self.map[y * self.width + x] = snake.head_value

    def valid_tiles(self, coord):
        dirs = []
        for direction in DIR_MAPPING:
            m_coord = coord_op(coord, direction, '+')
            x_move, y_move = m_coord
            if not self.is_inside(m_coord):
                continue
            if self.map[y_move * self.width + x_move] not in self.valid_tile_chars:
                continue
            dirs.append(m_coord)
        return dirs

    def update(self):
        start_time = time()
        self.time_step += 1
        self.map = self.fresh_map()
        self.alive_snakes = [s for h, s in self.snakes.items() if self.snakes_info[h]['alive']]
        alive_snakes = self.alive_snakes
        random.shuffle(alive_snakes)
        for snake in self.snakes.values():
            self.put_snake_on_map(snake)
        self.food.generate_new(self.map)
        self.print_map()
        for snake in alive_snakes:
            print(f"updating snake {snake.id}")
            self.snakes_info[snake.id]['old_tail'] = snake.body_coords[-1]
            self.snakes_info[snake.id]['last_coord'] = snake.coord
            self.snakes_info[snake.id]['length'] = snake.length
            next_coord = snake.update()
            if snake.alive:
                x, y = snake.coord
                self.snakes_info[snake.id]['current_coord'] = snake.coord
                if self.map[y * self.width + x] == self.FOOD_TILE:
                    self.snakes_info[snake.id]['last_food'] = self.time_step
                    self.food.remove_eaten(next_coord)
                    snake.length += 1
            else:
                self.snakes_info[snake.id]['alive'] = False
            self.update_snake_on_map(snake)
        # print(f"Total update time: {(time() - start_time) * 1000}")

    def get_color_map(self):
        color_map = {}
        for i, tile_value in enumerate(self.map):
            (y, x) = (i // self.width, i % self.width)
            if tile_value != self.FREE_TILE:
                color_map[(x, y)] = self.COLOR_MAPPING[tile_value]
        return color_map

    def get_flat_color_map(self):
        color_map = []
        for tile_value in self.map:
                color_map.append(self.COLOR_MAPPING[tile_value])
        return color_map


    def generate_run(self, out_file):
        steps = {}
        start_time = time()
        for snake in self.snakes.values():
            self.put_snake_on_map(snake)
        ongoing = True
        while ongoing:
            print(f"Step: {self.time_step}, passed time sec: {time() - start_time:.2f}")
            if self.alive_snakes:
                if len(self.alive_snakes) == 1:
                    only_one = self.alive_snakes[0]
                    if (self.time_step - self.snakes_info[only_one.id]['last_food']) > 100:
                        ongoing = False
                color_map = self.get_flat_color_map()
                steps[self.time_step] = {
                    'colors': color_map,
                    'state': copy.deepcopy(self.snakes_info)
                }
                self.update()
            else:
                ongoing = False
        print("GAME OVER")
        run_data = {
            'height': self.height,
            'width': self.width,
            'steps': steps,
            'total_time_s': time() - start_time
        }
        with open(out_file, 'w') as out_file:
            json.dump(run_data, out_file)
