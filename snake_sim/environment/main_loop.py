import random
import json
import numpy as np
from importlib import resources

from typing import Optional, List
from abc import ABC, abstractmethod

from snake_sim.utils import DotDict
from snake_sim.environment.snake_handlers import ISnakeHandler
from snake_sim.environment.snake_env import ISnakeEnv
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
    def set_snake_handler(self, snake_handler: ISnakeHandler):
        pass

    @abstractmethod
    def set_max_no_food_steps(self, steps):
        pass

    @abstractmethod
    def set_max_steps(self, steps):
        pass

    @abstractmethod
    def add_observer(self, observer: ILoopObserver):
        pass

    @abstractmethod
    def set_environment(self, env: ISnakeEnv):
        pass

class SimLoop(IMainLoop):

    def __init__(self):
        self._snake_handler: ISnakeHandler = None
        self._observers: List[ILoopObserver] = []
        self._env: ISnakeEnv = None
        self._max_no_food_steps = None
        self._max_steps = None
        self._steps = 0

    def start(self):
        while True:
            update_ordered_ids = self._snake_handler.get_update_order()
            for id in update_ordered_ids:
                decision = self._snake_handler.get_decision(id, self._env.get_env_data())
                alive = self._env.move_snake(id, decision)
                if not alive:
                    self._snake_handler.kill_snake(id)
                for observer in self._observers:
                    observer.notify(id, decision)
            self._steps += 1
            if self._max_no_food_steps and self._env.steps_since_any_ate() > self._max_no_food_steps:
                break
            if self._max_steps and self._steps > self._max_steps:
                break

    def set_snake_handler(self, snake_handler: ISnakeHandler):
        self._snake_handler = snake_handler

    def set_max_no_food_steps(self, steps):
        self._max_no_food_steps = steps

    def set_max_steps(self, steps):
        self._max_steps = steps

    def add_observer(self, observer: ILoopObserver):
        self._observers.append(observer)

    def set_environment(self, env: ISnakeEnv):
        self._env = env


class GameLoop(SimLoop):

    def __init__(self):
        self._steps_per_min = None

    def start(self):
        pass

    def set_steps_per_min(self, spm):
        self._steps_per_min = spm

