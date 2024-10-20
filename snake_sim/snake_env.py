import random
import array
import json
import os
import sys
from PIL import Image
import numpy as np
from typing import Optional, List
from collections import deque

from . import utils
from .utils import coord_op, exec_time, coord_cmp
from pathlib import Path
from .render import core as render
from .snakes.snake import Snake
from .render.core import put_snake_in_frame
from time import time
from dataclasses import dataclass, field
from snake_sim.snakes.snake import Snake

DIR_MAPPING = {
    (0, -1): 'up',
    (1,  0): 'right',
    (0,  1): 'down',
    (-1, 0): 'left'
}

class Food:

    def __init__(self, width: int, height: int, max_food: int, decay_count: Optional[int] = None):
        self.width = width
        self.height = height
        self.max_food = max_food
        self.decay_count = decay_count if decay_count else None
        self.locations = set()
        self.decay_counters = {}

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

    def clear(self):
        self.locations.clear()
        self.decay_counters.clear()


class StepData:
    def __init__(self, food: list, step: int) -> None:
        self.snakes: List[dict] = []
        self.food = food
        self.step = step

    @classmethod
    def from_dict(cls, step_dict):
        step_data = cls(food=step_dict['food'], step=step_dict['step'])
        for snake_data in step_dict['snakes']:
            step_data.snakes.append(snake_data)
        return step_data

    def add_snake_data(self, snake_coords: list, head_dir: tuple, tail_dir: tuple, snake_id: str):
        did_eat = False
        turn = None
        if snake_coords[0] in self.food:
            did_eat = True
        last_dir = coord_op(snake_coords[1], snake_coords[2], '-')
        if last_dir != head_dir:
            if last_dir == (0, -1):
                if head_dir == (1, 0):
                    turn = 'right'
                else:
                    turn = 'left'
            elif last_dir == (1, 0):
                if head_dir == (0, 1):
                    turn = 'right'
                else:
                    turn = 'left'
            elif last_dir == (0, 1):
                if head_dir == (-1, 0):
                    turn = 'right'
                else:
                    turn = 'left'
            elif last_dir == (-1, 0):
                if head_dir == (0, -1):
                    turn = 'right'
                else:
                    turn = 'left'

        self.snakes.append({
            'snake_id': snake_id,
            'curr_head': snake_coords[0],
            'prev_head': snake_coords[1] if len(snake_coords) > 1 else snake_coords[0],
            'curr_tail': snake_coords[-1],
            'head_dir': head_dir,
            'tail_dir': tail_dir,
            'did_eat': did_eat,
            'did_turn': turn
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
        self.food_value = SnakeEnv.FOOD_TILE
        self.free_value = SnakeEnv.FREE_TILE
        self.blocked_value = SnakeEnv.BLOCKED_TILE
        self.color_mapping = SnakeEnv.COLOR_MAPPING
        self.steps = {}

    @classmethod
    def from_dict(cls, run_dict):
        run_data = cls(
            width=run_dict['width'],
            height=run_dict['height'],
            snake_data=run_dict['snake_data'],
            base_map=np.array(run_dict['base_map'], dtype=np.uint8)
        )
        for step_nr, step_dict in run_dict['steps'].items():
            step_data_obj = StepData.from_dict(step_dict)
            run_data.add_step(int(step_nr), step_data_obj)
        run_data.food_value = run_dict['food_value']
        run_data.free_value = run_dict['free_value']
        run_data.blocked_value = run_dict['blocked_value']
        run_data.color_mapping = run_dict['color_mapping']
        return run_data

    @classmethod
    def from_json_file(cls, filepath):
        with open(filepath) as file:
            return cls.from_dict(json.load(file))

    def add_step(self, step: int, state: StepData):
        self.steps[step] = state

    def get_coord_mapping(self, step_nr):
        steps = self.steps
        final_step = len(steps)
        current = 1
        coords_map = {}
        last_step: StepData = steps[current]
        current_step: StepData = None
        while current <= step_nr < final_step:
            current_step = steps[current]
            snake_data = current_step.snakes
            for snake in snake_data:
                body = coords_map.setdefault(snake['snake_id'], deque([snake['prev_head']]))
                curr_head = snake['curr_head']
                body.appendleft(curr_head)
                if last_step is not None:
                    last_snake_step = list(filter(lambda x: x['snake_id'] == snake['snake_id'], last_step.snakes))[0]
                    if not coord_cmp(last_snake_step['curr_tail'], snake['curr_tail']):
                        body.pop()
            current += 1
            last_step = current_step
        if current_step is not None:
            coords_map['food'] = current_step.food
            return coords_map
        return None

    def write_to_file(self, aborted=False, ml=False, filepath=None):
        if filepath is None:
            runs_dir = Path(__file__).parent / self.output_dir
            run_dir = os.path.join(runs_dir, f'grid_{self.width}x{self.height}')
            os.makedirs(run_dir, exist_ok=True)
            aborted_str = '_ABORTED' if aborted else ''
            grid_str = f'{self.width}x{self.height}'
            nr_snakes = f'{len(self.snake_data)}'
            rand_str = utils.rand_str(6)
            filename = f'{nr_snakes}_snakes_{grid_str}_{rand_str}_{len(self.steps)}{"_ml_" if ml else ""}{aborted_str}.json'
            filepath = os.path.join(run_dir, filename)
        else:
            filepath = Path(filepath).absolute()

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
            'food_value': self.food_value,
            'free_value': self.free_value,
            'blocked_value': self.blocked_value,
            'color_mapping': self.color_mapping,
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

    @classmethod
    def get_map_files(self):
        map_dir = Path(__file__).parent / 'maps/map_images'
        maps = {}
        for f in map_dir.iterdir():
            if f.is_file() and f.suffix == '.png':
                maps[f.stem] = str(f)
        return maps

    def load_png_map(self, map_path):
        if not Path(map_path).is_absolute():
            maps = self.get_map_files()
            try:
                img_path = Path(maps[map_path])
            except KeyError:
                raise ValueError(f"Map with name '{map_path}' not found")
        else:
            img_path = Path(map_path)
        try:
            image = Image.open(img_path)
        except FileNotFoundError:
            raise ValueError(f"Map file not found: {img_path}")
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
                    self.base_map[y, x] = map_color_mapping[color]
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
        self.food.clear()

    def print_map(self, s_map=None):
        if s_map is None:
            s_map = self.map
        print(f"{'':@<{self.width*3}}")
        for y in range(self.height):
            print_row = []
            for c in s_map[y]:
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

    def print_stats(self):
        for snake_info in self.snakes_info.values():
            print(f"Snake: {snake_info['id']}, length: {snake_info['length']}")

    def add_snake(self, snake, h_color, b_color):
        if isinstance(snake, Snake):
            snake.bind_env(self)
            while True:
                rand_x = round(random.random() * (self.width - 1))
                rand_y = round(random.random() * (self.height - 1))
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

    def update(self, verbose=True):
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
            direction = snake.update()
            next_coord = coord_op(snake.coord, direction, '+')
            # print(direction, next_coord, snake.coord, snake.body_coords[1])
            if next_coord not in self.valid_tiles(snake.coord):
                snake.kill()
            else:
                snake.set_new_head(next_coord)
            if verbose:
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
        if verbose:
            print(f"Step: {self.time_step}, {len(self.alive_snakes)} alive")

    def get_expanded_normal_map(self, snake_id, expand_factor=2):
        new_height, new_width = coord_op(self.map.shape, (expand_factor, expand_factor), '*')
        expanded_map = np.full((new_height, new_width), self.FREE_TILE, dtype=np.uint8)
        for snake in self.snakes.values():
            if snake.id != snake_id:
                put_snake_in_frame(expanded_map, snake.body_coords, self.NORM_BODY, expand_factor)
                expanded_map[snake.y * expand_factor, snake.x * expand_factor] = self.NORM_HEAD
            else:
                put_snake_in_frame(expanded_map, snake.body_coords, self.NORM_MAIN_BODY, expand_factor)
                expanded_map[snake.y * expand_factor, snake.x * expand_factor] = self.NORM_MAIN_HEAD
        expanded_map = expanded_map.reshape(new_height, new_width, 1)
        return expanded_map

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
                        if any(snake_data['alive'] for snake_data in self.snakes_info.values()):
                            conn.send(self.run_data.steps[self.time_step].to_dict())
                    else:
                        ongoing = False
                        # break

                except KeyboardInterrupt:
                    aborted = True
                    sys.exit(0)
        finally:
            print('Done')
            self.print_stats()
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

    def ml_training_run(self, episodes, start_episode=0, max_steps=None, max_no_food_steps=200):
        for episode in range(start_episode, episodes + start_episode):
            self.init_recorder()
            # print("Episode: ", episode)
            self.reset()
            for snake in self.snakes.values():
                self.put_snake_on_map(snake)
            ongoing = True
            aborted = False
            try:
                while ongoing:
                    if self.alive_snakes:
                        highest_no_food = max([self.snakes_info[only_one.id]['last_food'] for only_one in self.alive_snakes])
                        if (self.time_step - highest_no_food) > max_no_food_steps or max_steps is not None and self.time_step > max_steps:
                            ongoing = False
                        self.update(verbose=False)
                    else:
                        ongoing = False
                    # self.print_map()

            except KeyboardInterrupt:
                print("Keyboard interrupt detected")
                for snake in self.snakes.values():
                    try:
                        snake.save_weights()
                    except AttributeError:
                        pass
                aborted = True
            finally:
                # self.run_data.write_to_file(aborted=aborted, ml=True)
                if aborted:
                    raise KeyboardInterrupt
            for snake in self.snakes.values():
                try:
                    pass
                    # print(f'Episode: {episode} total reward: {snake.total_reward}')
                except AttributeError:
                    pass