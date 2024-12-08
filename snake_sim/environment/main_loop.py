import random
import json
import importlib.resources as pkg_resources
import numpy as np

from PIL import Image
from typing import Optional, List
from abc import ABC, abstractmethod
from pathlib import Path

from snake_sim.utils import DotDict
from snake_sim.environment.snake_handlers import ISnakeHandler
from snake_sim.environment.food_handlers import FoodHandler

with pkg_resources.open_text('snake_sim.config', 'default_config.json') as config_file:
    config = DotDict(json.load(config_file))


class ILoopObserver(ABC):

    @abstractmethod
    def notify(self, *args, **kwargs):
        pass


class IMainLoop(ABC):

    @abstractmethod
    def init(self, width, height):
        pass

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def add_snake_handler(self, snake_handler: ISnakeHandler):
        pass

    @abstractmethod
    def add_observer(self, observer: ILoopObserver):
        pass

    @abstractmethod
    def set_food_handler(self, food_handler: FoodHandler):
        pass

    @abstractmethod
    def load_map(self, map_img_path):
        pass

    @abstractmethod
    def resize(self, width, height):
        pass


class SimLoop(IMainLoop):

    def __init__(self):
        self._snake_handler: ISnakeHandler = None
        self._food_handler: FoodHandler = None
        self._observers: List[ILoopObserver] = []
        self._width = None
        self._height = None
        self._map = None

    def init(self, width, height):
        self._width = width
        self._height = height
        self._map = np.full((height, width), config.FREE_TILE, dtype=np.uint8)

    def start(self):
        pass

    def add_snake_handler(self, snake_handler: ISnakeHandler):
        self._snake_handler = snake_handler

    def set_food_handler(self, food_handler: FoodHandler):
        self._food_handler = food_handler

    def fresh_map(self):
        return np.copy(self._map)

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

    def resize(self, width, height):
        self._width = width
        self._height = height
        self._map = np.full((height, width), config.FREE_TILE, dtype=np.uint8)
        self._food_handler.resize(width, height)


class GameLoop(SimLoop):

    def __init__(self):
        self._steps_per_min = None

    def start(self):
        pass

    def set_steps_per_min(self, spm):
        self._steps_per_min = spm

