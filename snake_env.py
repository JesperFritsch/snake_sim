import random
import array
import json
import os

import utils
from snakes.snake import Snake
from time import time
from dataclasses import dataclass, field


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

@dataclass
class Food:

    width: int
    height: int
    max_food: int
    decay_count: int = 100
    locations: set = field(default_factory=set)
    decay_counters: dict = field(default_factory=dict)

    def generate_new(self, s_map) -> list:
        empty_tiles = []
        for i in range(self.width * self.height):
            if s_map[i] == SnakeEnv.FREE_TILE:
                x, y = i % self.width, i // self.width
                empty_tiles.append((x, y))
        for _ in range(len(self.locations), self.max_food):
            if empty_tiles:
                new_food = random.choice(empty_tiles)
                empty_tiles.remove(new_food)
                self.add_new(new_food)
        for location in list(self.locations):
            self.decay_counters[location] -= 1
            if self.decay_counters[location] <= 0:
                self.remove_eaten(location)
        for location in self.locations:
            x, y = location
            s_map[y * self.width + x] = SnakeEnv.FOOD_TILE


    def add_new(self, coord):
        self.decay_counters[coord] = self.decay_count
        self.locations.add(coord)

    def remove_eaten(self, coord):
        if coord in self.locations:
            self.locations.remove(coord)


class StepData:
    def __init__(self, food: list, step: int) -> None:
        self.snakes = []
        self.food = food
        self.step = step

    def add_snake_data(self, snake_coords: list, head_dir: tuple, tail_dir: tuple, snake_id: str):
        self.snakes.append({
            'snake_id': snake_id,
            'coords': snake_coords,
            'head_dir': head_dir,
            'tail_dir': tail_dir
        })

    def to_dict(self):
        return {
            'snakes': self.snakes,
            'food': self.food,
            'step': self.step
        }


class RunData:
    def __init__(self, width: int, height: int, snake_data: list, output_dir='runs') -> None:
        self.width = width
        self.height = height
        self.snake_data = snake_data
        self.output_dir = output_dir
        self.steps = {}

    def add_step(self, step: int, state: StepData):
        self.steps[step] = state

    def write_to_file(self, aborted=False):
        runs_dir = os.path.abspath(os.path.join(os.getcwd(), self.output_dir))
        run_dir = os.path.join(runs_dir, f'grid_{self.width}x{self.height}')
        os.makedirs(run_dir, exist_ok=True)
        aborted_str = '_ABORTED' if aborted else ''
        grid_str = f'{self.width}x{self.height}'
        nr_snakes = f'{len(self.snake_data)}'
        filename = f'{nr_snakes}_snakes_{grid_str}_{utils.rand_str(6)}{aborted_str}.json'
        filepath = os.path.join(run_dir, filename)
        print(f"saving run data to '{filepath}'")
        with open(filepath, 'w') as file:
            json.dump(self.to_dict(), file)

    def to_dict(self):
        return {
            'width': self.width,
            'height': self.height,
            'snake_data': self.snake_data,
            'steps': {k: v.to_dict() for k, v in self.steps.items()}
        }


class SnakeEnv:
    valid_tile_values = (FOOD_TILE, FREE_TILE) = (70, 46)
    COLOR_MAPPING = {
        FOOD_TILE: (223, 163, 49),
        FREE_TILE: (0, 0, 0)
    }

    def __init__(self, width, height, food, output_dir='runs') -> None:
        self.map = None
        self.snakes = {}
        self.width = width
        self.height = height
        self.time_step = 0
        self.alive_snakes = []
        self.map = self.fresh_map()
        self.snakes_info = {}
        self.run_data = None
        self.food = Food(width=width, height=height, max_food=food)

    def fresh_map(self):
        return array.array('B', [self.FREE_TILE] * (self.width * self.height))

    def reset(self):
        self.time_step = 0
        self.map = self.fresh_map()
        for snake in self.snakes.values():
            snake.reset()
            self.snakes_info[snake.id].update({
                    'length': snake.length,
                    'head_dir': (0,0),
                    'tail_dir': (0,0),
                    'alive': True,
                    'id': snake.id.upper(),
                    'last_food': 0
                })
            rand_x = round(random.random() * (self.width - 1)) - 1
            rand_y = round(random.random() * (self.height - 1)) - 1
            snake.set_init_coord((rand_x, rand_y))
        self.alive_snakes = self.get_alive_snakes()

    def print_map(self):
        print(f"{'':@<{self.width*3}}")
        for y in range(self.height):
            print(''.join([f' {chr(c)} ' for c in self.map[y*self.width:y*self.width+self.width]]))

    def init_recorder(self):
        self.run_data = RunData(
            width=self.width,
            height=self.height,
            snake_data=[{
                'snake_id': s.id,
                'head_color': self.COLOR_MAPPING[s.head_value],
                'body_color': self.COLOR_MAPPING[s.body_value]
                } for s in self.snakes.values()])

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
                    'head_dir': (0,0),
                    'tail_dir': (0,0),
                    'alive': True,
                    'id': snake.id.upper(),
                    'head_value': head_value,
                    'body_value': body_value,
                    'h_color': h_color,
                    'b_color': b_color,
                    'last_food': 0
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

    def get_alive_snakes(self):
        return [s for h, s in self.snakes.items() if self.snakes_info[h]['alive']]

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
        self.time_step += 1
        self.map = self.fresh_map() # needed for the snakes, without it the snakes map is never cleared.
        self.alive_snakes = self.get_alive_snakes()
        alive_snakes = self.alive_snakes
        random.shuffle(alive_snakes)
        for snake in self.snakes.values():
            self.put_snake_on_map(snake)
        self.food.generate_new(self.map)
        new_step = StepData(food=list(self.food.locations), step=self.time_step)
        for snake in alive_snakes:
            old_tail = snake.body_coords[-1]
            self.snakes_info[snake.id]['length'] = snake.length
            next_coord = snake.update()
            if snake.alive:
                x, y = snake.coord
                self.snakes_info[snake.id]['current_coord'] = snake.coord
                self.snakes_info[snake.id]['head_dir'] = coord_op(snake.coord, snake.body_coords[1], '-')
                self.snakes_info[snake.id]['tail_dir'] = coord_op(snake.body_coords[-1], old_tail, '-')
                if self.map[y * self.width + x] == self.FOOD_TILE:
                    self.snakes_info[snake.id]['last_food'] = self.time_step
                    self.food.remove_eaten(next_coord)
                    snake.length += 1
            else:
                self.snakes_info[snake.id]['alive'] = False
            new_step.add_snake_data(
                list(snake.body_coords),
                self.snakes_info[snake.id]['head_dir'],
                self.snakes_info[snake.id]['tail_dir'],
                snake.id)
            self.update_snake_on_map(snake)
        self.run_data.add_step(self.time_step, new_step)

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

    def generate_run(self, max_steps: int|None = None):
        start_time = time()
        self.init_recorder()
        for snake in self.snakes.values():
            self.put_snake_on_map(snake)
        ongoing = True
        aborted = False
        try:
            while ongoing:
                print(f"Step: {self.time_step}, passed time sec: {time() - start_time:.2f}")
                if self.alive_snakes:
                    if len(self.alive_snakes) == 1:
                        only_one = self.alive_snakes[0]
                        if (self.time_step - self.snakes_info[only_one.id]['last_food']) > 100 or max_steps is not None and self.time_step > max_steps:
                            ongoing = False
                    self.update()
                else:
                    ongoing = False
        except KeyboardInterrupt:
            print("Keyboard interrupt detected")
            aborted = True
        finally:
            print('Done')
            self.run_data.write_to_file(aborted=aborted)
            if aborted:
                raise KeyboardInterrupt
