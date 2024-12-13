import json
import numpy as np
from typing import Optional, Iterable, Dict
from abc import ABC, abstractmethod
from PIL import Image
from pathlib import Path
from importlib import resources
from collections import deque

from snake_sim.utils import DotDict, coord_op

with resources.open_text('snake_sim.config', 'default_config.json') as config_file:
    config = DotDict(json.load(config_file))


class SnakeRep:
    def __init__(self, id: int):
        self.move_count = 0
        self.last_ate = 0
        self.id = id
        self.body = deque()

    def set_body(self, body: Iterable[tuple]):
        self.body = deque(body)

    def move(self, direction: tuple, grow=False):
        self.body.appendleft(coord_op(self.body[0], direction, '+'))
        if not grow:
            self.last_ate = self.move_count
            self.body.pop()
        self.move_count += 1

    def get_head(self):
        return self.body[0]

    def get_tail(self):
        return self.body[-1]


class ISnakeEnv(ABC):
    @abstractmethod
    def get_map(self, id: Optional[int]):
        pass

    @abstractmethod
    def move_snake(self, id: int, direction: tuple):
        pass

    @abstractmethod
    def load_map(self, map_img_path: str):
        pass

    @abstractmethod
    def add_snake(self, id: int, snake_body: Iterable[tuple]):
        pass


class SnakeEnv(ISnakeEnv):
    def __init__(self, width, height):
        self._width = width
        self._height = height
        self._map = np.full((height, width), config.FREE_TILE, dtype=np.uint8)
        self._snake_reps: Dict[int, SnakeRep] = {}

    def add_snake(self, id: int, snake_body: Iterable[tuple]):
        # snake_body is expected to be an iterable with tuples of coords like [(1,2), (1,3)]
        snake = SnakeRep(id)
        self._snake_reps[id] = snake
        snake.set_body(snake_body)
        self._place_snake_on_map(snake)

    def _place_snake_on_map(self, snake_rep: SnakeRep):
        for i, (x, y) in enumerate(snake_rep.body):
            self._map[y, x] = self._snake_reps.id + 0 if i == 0 else 1

    def _remove_snake_from_map(self, snake_rep: SnakeRep):
        for x, y in snake_rep.body:
            self._map[y, x] = config.FREE_TILE

    def move_snake(self, id: int, direction: tuple):
        snake_rep = self._snake_reps[id]
        next_tile = coord_op(snake_rep.get_head(), direction, '+')
        if direction not in config.DIRS.values() or not self.free_tile(next_tile):
            return False
        old_tail = snake_rep.body[-1]
        snake_rep.move(direction)
        new_tail = snake_rep.body[-1]
        if old_tail != new_tail:
            self._map[old_tail[1], old_tail[0]] = config.FREE_TILE

    def free_tile(self, coord: tuple):
        x, y = coord
        return self._map[y, x] <= config.FREE_TILE

    def fresh_map(self):
        return np.copy(self._map)

    def get_map(self, id: Optional[int] = None):
        return self.fresh_map()

    def load_map(self, map_img_path: str):
        img_path = Path(map_img_path)
        image = Image.open(img_path)
        self.resize(*image.size)
        image_matrix = np.array(image)
        map_color_mapping = {
            (0,0,0,0): config.FREE_TILE,
            (255,0,0,255): config.FOOD_TILE,
            (0,0,0,255): config.BLOCKED_TILE
        }
        for y in range(self._height):
            for x in range(self._width):
                color = tuple(image_matrix[y][x])
                try:
                    self._map[y, x] = map_color_mapping[color]
                except KeyError:
                    print(f"Color '{color}' at (x={x}, y={y}) from image not found in color mapping")