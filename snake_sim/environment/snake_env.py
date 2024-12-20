import json
import numpy as np
from typing import Optional, Dict, Deque
from PIL import Image
from pathlib import Path
from importlib import resources
from collections import deque

from snake_sim.utils import DotDict, Coord
from snake_sim.environment.food_handlers import IFoodHandler
from snake_sim.environment.interfaces.snake_env_interface import ISnakeEnv

with resources.open_text('snake_sim.config', 'default_config.json') as config_file:
    config = DotDict(json.load(config_file))


class SnakeRep:
    def __init__(self, id: int, start_position: Coord, start_length: int=1):
        self.move_count = 0
        self.last_ate = 0
        self.id = id
        self.body: Deque[Coord] = deque([start_position] * start_length)
        self.is_alive = True

    def kill(self):
        self.is_alive = False

    def move(self, direction: Coord, grow=False):
        self.body.appendleft(self.get_head() + direction)
        if not grow:
            self.last_ate = self.move_count
            self.body.pop()
        self.move_count += 1

    def get_head(self) -> Coord:
        return self.body[0]

    def get_tail(self) -> Coord:
        return self.body[-1]


class EnvData:
    def __init__(self, map: np.ndarray, snakes: Dict[int, SnakeRep]):
        self.map = map.tobytes()
        self.snakes = {id: {
                                'is_alive': snake_rep.is_alive,
                                'length': len(snake_rep.body),
                            } for id, snake_rep in snakes.items()}


class EnvInitData:
    def __init__(self,
                height: int,
                width: int,
                free_value: int,
                blocked_value: int,
                food_value: int,
                snake_reps: Dict[int, SnakeRep],
                base_map: np.ndarray):
        self.height = height
        self.width = width
        self.free_value = free_value
        self.blocked_value = blocked_value
        self.food_value = food_value
        self.snake_values = {id: {
            "head_value": id,
            "body_value": id + 1,
        } for id in snake_reps.keys()}
        self.start_positions = {id: snake_rep.get_head() for id, snake_rep in snake_reps.items()}
        self.base_map = base_map


class SnakeEnv(ISnakeEnv):
    def __init__(self, width, height, free_value, blocked_value, food_value):
        self._width = width
        self._height = height
        self._free_value = free_value
        self._blocked_value = blocked_value
        self._food_value = food_value
        self._map = np.full((height, width), self._free_value, dtype=np.uint8)
        self._base_map = np.copy(self._map)
        self._snake_reps: Dict[int, SnakeRep] = {}

    def add_snake(self, id: int, start_position: Optional[Coord]=None, start_length: int=1) -> Coord:
        # snake_body is expected to be an iterable with Coord like [(1,2), (1,3)]
        if start_position is None:
            start_position = self._random_free_tile()
        snake_rep = SnakeRep(id, start_position, start_length)
        self._snake_reps[id] = snake_rep
        self._place_snake_on_map(snake_rep)
        return start_position

    def _random_free_tile(self) -> Coord:
        while True:
            rand_coord = Coord(np.random.randint(1, self._width - 1), np.random.randint(1, self._height - 1))
            x, y = rand_coord
            if self._map[y, x] == self._free_value:
                return rand_coord

    def _place_snake_on_map(self, snake_rep: SnakeRep):
        for i, (x, y) in enumerate(snake_rep.body):
            self._map[y, x] = snake_rep.id + (0 if i == 0 else 1)

    def _remove_snake_from_map(self, snake_rep: SnakeRep):
        for x, y in snake_rep.body:
            self._map[y, x] = self._free_value

    def move_snake(self, id: int, direction: Coord):
        # direction is expected to be a Coord like (1, 0) for right
        # returns False if the move is invalid, otherwise True
        snake_rep = self._snake_reps[id]
        current_head = snake_rep.get_head()
        next_tile = current_head + direction
        if direction not in config.DIRS.values() or not self._free_tile(next_tile):
            return False, False

        grow = False
        if self._map[next_tile[1], next_tile[0]] == self._food_value:
            self._food_handler.remove(next_tile, self._map)
            grow = True
        old_tail = snake_rep.body[-1]
        snake_rep.move(direction, grow)
        new_tail = snake_rep.body[-1]
        self._map[next_tile[1], next_tile[0]] = id # set the tile to the snakes head value
        self._map[current_head[1], current_head[0]] = id + 1 # set the old head tile to the snakes body value
        if old_tail != new_tail:
            self._map[old_tail[1], old_tail[0]] = self._free_value
        return True, grow

    def _free_tile(self, coord: Coord):
        x, y = coord
        return 0 <= x < self._width and 0 <= y < self._height and self._map[y, x] <= self._free_value

    def get_map(self):
        return np.copy(self._map)

    def get_base_map(self):
        return np.copy(self._base_map)

    def get_env_data(self, id: Optional[int] = None) -> EnvData:
        # id is not used yet, but it is preparing for being able to send different data to different snakes
        return EnvData(self.get_map(), self._snake_reps)

    def get_food(self):
        return self._food_handler.get_food()

    def get_init_data(self) -> EnvInitData:
        return EnvInitData(
            self._height,
            self._width,
            self._free_value,
            self._blocked_value,
            self._food_value,
            self._snake_reps,
            self.get_base_map())

    def resize(self, height, width):
        self._height = height
        self._width = width
        self._map.resize((height, width))
        self._food_handler.resize(height, width)

    def load_map(self, map_img_path: str):
        img_path = Path(map_img_path)
        image = Image.open(img_path)
        self.resize(*image.size)
        image_matrix = np.array(image)
        map_color_mapping = {
            (0,0,0,0): self._free_value,
            (255,0,0,255): self._food_value,
            (0,0,0,255): self._blocked_value
        }
        for y in range(self._height):
            for x in range(self._width):
                color = tuple(image_matrix[y][x])
                try:
                    value = map_color_mapping[color]
                    self._base_map[y, x] = value
                    if value == self._food_value:
                        self._food_handler.add_new((x, y))
                except KeyError:
                    print(f"Color '{color}' at (x={x}, y={y}) from image not found in color mapping")
        self._map = self.get_base_map()

    def set_food_handler(self, food_handler: IFoodHandler):
        if not isinstance(food_handler, IFoodHandler):
            raise ValueError("food_handler must be an instance of IFoodHandler")
        self._food_handler = food_handler
        self._food_handler.resize(self._width, self._height)

    def update_food(self):
        self._food_handler.update(self._map)

    def steps_since_any_ate(self):
        return min(snake_rep.move_count - snake_rep.last_ate for snake_rep in self._snake_reps.values())

    def print_map(self):
        for row in self._map:
            print_row = []
            for c in row:
                if c == self._free_value:
                    print_row.append(' . ')
                elif c == self._food_value:
                    print_row.append(' F ')
                elif c == self._blocked_value:
                    print_row.append(' # ')
                elif c % 2 == 0:
                    print_row.append(f' X ')
                else:
                    print_row.append(f' x ')
            print(''.join(print_row))
