import random
import json
import numpy as np
from importlib import resources

from typing import Optional, List
from abc import ABC, abstractmethod

from snake_sim.utils import DotDict
from snake_sim.environment.snake_handlers import ISnakeHandler
from snake_sim.environment.food_handlers import FoodHandler

with resources.open_text('snake_sim.config', 'default_config.json') as config_file:
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


class GameLoop(SimLoop):

    def __init__(self):
        self._steps_per_min = None

    def start(self):
        pass

    def set_steps_per_min(self, spm):
        self._steps_per_min = spm

