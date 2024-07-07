import random
import array
import json
import os
import sys
from PIL import Image
import numpy as np
from typing import Optional

import utils
from utils import coord_op, exec_time
from pathlib import Path
from render import core as render
from snakes.snake import Snake
from time import time
from dataclasses import dataclass, field

DIR_MAPPING = {
    (0, -1): 'up',
    (1,  0): 'right',
    (0,  1): 'down',
    (-1, 0): 'left'
}

@dataclass
class Food:

    width: int
    height: int
    max_food: int
    decay_count: Optional[int] = None
    locations: set = field(default_factory=set)
    decay_counters: dict = field(default_factory=dict)

    def generate_new(self, s_map):
        empty_tiles = []
        for y in range(self.height):
            for x in range(self.width):
                if s_map[y, x] == SnakeEnv.FREE_TILE:
                    empty_tiles.append((x, y))
        for _ in range(self.max_food - len(self.locations)):
            if empty_tiles:
                new_food = random.choice(empty_tiles)
                empty_tiles.remove(new_food)
                self.add_new(new_food)
        for location in self.locations:
            x, y = location
            s_map[y, x] = SnakeEnv.FOOD_TILE

    def remove_old(self, s_map):
        if self.decay_count is None:
            return
        for location in set(self.locations):
            self.decay_counters[location] -= 1
            if self.decay_counters[location] <= 0:
                self.remove(location, s_map)

    def add_new(self, coord):
        self.decay_counters[coord] = self.decay_count
        self.locations.add(coord)

    def remove(self, coord, s_map):
        if coord in self.locations:
            x, y = coord
            s_map[y, x] = SnakeEnv.FREE_TILE
            del self.decay_counters[coord]
            self.locations.remove(coord)


class StepData:
    def __init__(self, food: list, step: int) -> None:
        self.snakes = []
        self.food = food
        self.step = step

    def add_snake_data(self, snake_coords: list, head_dir: tuple, tail_dir: tuple, snake_id: str):
        self.snakes.append({
            'snake_id': snake_id,
            'curr_head': snake_coords[0],
            'prev_head': snake_coords[1],
            'curr_tail': snake_coords[-1],
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
    def __init__(self, width: int, height: int, snake_data: list, base_map: np.array,  output_dir='runs') -> None:
        self.width = width
        self.height = height
        self.snake_data = snake_data
        self.base_map = base_map
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
        rand_str = utils.rand_str(6)
        filename = f'{nr_snakes}_snakes_{grid_str}_{rand_str}_{len(self.steps)}{aborted_str}.json'
        filepath = os.path.join(run_dir, filename)
        print(f"saving run data to '{filepath}'")
        with open(filepath, 'w') as file:
            json.dump(self.to_dict(), file)
        pixel_changes = render.pixel_changes_from_runfile(filepath, 2)
        if self.width == 32 and self.height == 32:
            rpi_runs_dir = os.path.join(run_dir, 'rpi')
            os.makedirs(rpi_runs_dir, exist_ok=True)
            rpi_filepath = os.path.join(rpi_runs_dir, f'{nr_snakes}_snakes_rpi_{rand_str}_{len(self.steps)}{aborted_str}.run')
            with open(rpi_filepath, 'w') as file:
                for change in pixel_changes:
                    file.write(json.dumps(change) + '\n')

    def to_dict(self):
        return {
            'width': self.width,
            'height': self.height,
            'food_value': SnakeEnv.FOOD_TILE,
            'free_value': SnakeEnv.FREE_TILE,
            'blocked_value': SnakeEnv.BLOCKED_TILE,
            'color_mapping': SnakeEnv.COLOR_MAPPING,
            'snake_data': self.snake_data,
            'base_map': self.base_map.tolist(),
            'steps': {k: v.to_dict() for k, v in self.steps.items()}
        }


class SnakeEnv:
    FOOD_TILE = 0
    FREE_TILE = 1
    BLOCKED_TILE = 2
    NORM_HEAD = 3
    NORM_BODY = 4
    NORM_MAIN_HEAD = 5
    NORM_MAIN_BODY = 6
    valid_tile_values = (FOOD_TILE, FREE_TILE)
    COLOR_MAPPING = {
        FOOD_TILE: (223, 163, 49),
        FREE_TILE: (0, 0, 0),
        BLOCKED_TILE: (127, 127, 127)
    }

    def __init__(self, width, height, food, food_decay=500) -> None:
        self.snakes = {}
        self.width = width
        self.height = height
        self.time_step = 0
        self.alive_snakes = []
        self.base_map = np.full((height, width), self.FREE_TILE, dtype=np.uint8)
        self.map = self.fresh_map()
        self.snakes_info = {}
        self.run_data = None
        self.store_runs = True
        self.food = Food(width=width, height=height, max_food=food, decay_count=food_decay)

    def fresh_map(self):
        return self.base_map.copy()

    def load_png_map(self, map_path):
        img_path = Path(map_path)
        if not img_path.is_absolute():
            img_path = Path(__file__).parent.joinpath('maps/map_images') / img_path
        image = Image.open(img_path)
        image_matrix = np.array(image)
        map_color_mapping = {
            (0,0,0,0): self.FREE_TILE,
            (255,0,0,255): self.FOOD_TILE,
            (0,0,0,255): self.BLOCKED_TILE
        }
        for y in range(self.height):
            for x in range(self.width):
                color = tuple(image_matrix[y][x])
                try:
                    self.base_map[y * self.width + x] = map_color_mapping[color]
                except KeyError:
                    print(f"Color at (x={x}, y={y}) not found in color mapping: {color}")
        self.map = self.fresh_map()

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
            rand_x = 2 + round(random.random() * (self.width - 3))
            rand_y = 2 + round(random.random() * (self.height - 3))
            snake.set_init_coord((rand_x, rand_y))
        self.alive_snakes = self.get_alive_snakes()

    def print_map(self):
        print(f"{'':@<{self.width*3}}")
        for y in range(self.height):
            print_row = []
            for c in self.map[y]:
                if c == self.FREE_TILE:
                    print_row.append(' . ')
                elif c == self.FOOD_TILE:
                    print_row.append(' F ')
                elif c == self.BLOCKED_TILE:
                    print_row.append(' # ')
                else:
                    print_row.append(f' {chr(c)} ')
            print(''.join(print_row))

    def init_recorder(self):
        self.run_data = RunData(
            width=self.width,
            height=self.height,
            snake_data=[{
                'snake_id': s.id,
                'head_color': self.COLOR_MAPPING[s.head_value],
                'body_color': self.COLOR_MAPPING[s.body_value]
                } for s in self.snakes.values()],
            base_map=np.array(self.base_map))

    def add_snake(self, snake, h_color, b_color):
        if isinstance(snake, Snake):
            snake.bind_env(self)
            while True:
                rand_x = 2 + round(random.random() * (self.width - 3))
                rand_y = 2 + round(random.random() * (self.height - 3))
                if not self.map[rand_y, rand_x] == self.FREE_TILE:
                    continue
                break
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
            self.map[o_y, o_x] = self.FREE_TILE
        for i in range(2):
            x, y = snake.body_coords[i]
            self.map[y, x] = snake.head_value if (x, y) == head else snake.body_value

    def put_snake_on_map(self, snake):
        for coord in snake.body_coords:
            x, y = coord
            self.map[y, x] = snake.body_value
        x, y = snake.coord
        self.map[y, x] = snake.head_value

    def put_coords_on_map(self, coords, value):
        for coord in coords:
            x, y = coord
            self.map[y, x] = value

    def valid_tiles(self, coord, discount=None):
        tiles = []
        for direction in DIR_MAPPING:
            m_coord = coord_op(coord, direction, '+')
            x_move, y_move = m_coord
            if m_coord == discount:
                tiles.append(m_coord)
            elif not self.is_inside(m_coord):
                continue
            elif self.map[y_move, x_move] not in self.valid_tile_values:
                continue
            tiles.append(m_coord)
        return tiles

    def update(self):
        self.time_step += 1
        self.map = self.fresh_map() # needed for the snakes, without it the snakes map is never cleared.
        self.alive_snakes = self.get_alive_snakes()
        alive_snakes = self.alive_snakes
        random.shuffle(alive_snakes)
        for snake in self.snakes.values():
            self.put_snake_on_map(snake)
        self.food.generate_new(self.map)
        self.food.remove_old(self.map)
        # self.print_map()
        new_step = StepData(food=list(self.food.locations), step=self.time_step)
        for snake in alive_snakes:
            old_tail = snake.body_coords[-1]
            self.snakes_info[snake.id]['length'] = snake.length
            u_time = time()
            next_coord = snake.update()
            if next_coord not in self.valid_tiles(snake.coord):
                snake.kill()
            else:
                snake.set_new_head(next_coord)
            print(f'update_time for snake: {snake.id}', time() - u_time)
            if snake.alive:
                x, y = snake.coord
                self.snakes_info[snake.id]['current_coord'] = snake.coord
                self.snakes_info[snake.id]['head_dir'] = coord_op(snake.coord, snake.body_coords[1], '-')
                self.snakes_info[snake.id]['tail_dir'] = coord_op(snake.body_coords[-1], old_tail, '-')
                if self.map[y, x] == self.FOOD_TILE:
                    self.snakes_info[snake.id]['last_food'] = self.time_step
                    self.food.remove(next_coord, self.map)
            else:
                self.snakes_info[snake.id]['alive'] = False
            new_step.add_snake_data(
                list(snake.body_coords),
                self.snakes_info[snake.id]['head_dir'],
                self.snakes_info[snake.id]['tail_dir'],
                snake.id)
            self.update_snake_on_map(snake)
        self.run_data.add_step(self.time_step, new_step)
        print(f"Step: {self.time_step}, {len(self.alive_snakes)} alive")

    def get_normalized_map(self, snake_id):
        """ Used by ML models to get the map in a normalized format """
        norm_map = self.map.copy()
        for snake_info in self.snakes_info.values():
            if snake_info['id'] != snake_id:
                head_value = snake_info['head_value']
                body_value = snake_info['body_value']
                norm_map[norm_map == head_value] = self.NORM_HEAD
                norm_map[norm_map == body_value] = self.NORM_BODY
            else:
                head_value = snake_info['head_value']
                body_value = snake_info['body_value']
                norm_map[norm_map == head_value] = self.NORM_MAIN_HEAD
                norm_map[norm_map == body_value] = self.NORM_MAIN_BODY


    def stream_run(self, conn, max_steps=None, max_no_food_steps=500):
        self.init_recorder()
        for snake in self.snakes.values():
            self.put_snake_on_map(snake)
        ongoing = True
        aborted = False
        #send init data
        init_data = self.run_data.to_dict()
        del init_data['steps']
        conn.send(init_data)
        #wait for init ack
        try:
            while ongoing:
                try:
                    if conn.poll():
                        data = conn.recv()
                        if data == 'stop':
                            conn.send('stopped')
                            ongoing = False
                            aborted = True
                            break
                    if self.alive_snakes:
                        highest_no_food = max([self.snakes_info[only_one.id]['last_food'] for only_one in self.alive_snakes])
                        if (self.time_step - highest_no_food) > max_no_food_steps or max_steps is not None and self.time_step > max_steps:
                            ongoing = False
                        self.update()
                        conn.send(self.run_data.steps[self.time_step].to_dict())
                    else:
                        ongoing = False
                        # break

                except KeyboardInterrupt:
                    aborted = True
                    sys.exit(0)
        finally:
            print('Done')
            if self.store_runs:
                self.run_data.write_to_file(aborted=aborted)

    def generate_run(self, max_steps=None, max_no_food_steps=500):
        start_time = time()
        self.init_recorder()
        for snake in self.snakes.values():
            self.put_snake_on_map(snake)
        ongoing = True
        aborted = False
        try:
            while ongoing:
                print(f"Step: {self.time_step}, passed time sec: {time() - start_time:.2f} {len(self.alive_snakes)} alive")
                if self.alive_snakes:
                    highest_no_food = max([self.snakes_info[only_one.id]['last_food'] for only_one in self.alive_snakes])
                    if (self.time_step - highest_no_food) > max_no_food_steps or max_steps is not None and self.time_step > max_steps:
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
