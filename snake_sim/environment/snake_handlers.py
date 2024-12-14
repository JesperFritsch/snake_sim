import random
from abc import ABC, abstractmethod
from typing import Optional, Dict

from snake_sim.snakes.snake import ISnake
from snake_sim.environment.snake_env import EnvData
from snake_sim.utils import Coord

class ISnakeHandler(ABC):
    @abstractmethod
    def get_decision(self, id, env_data: EnvData) -> Coord:
        pass

    @abstractmethod
    def add_snake(self, snake: ISnake):
        pass

    @abstractmethod
    def get_update_order(self):
        pass

    @abstractmethod
    def kill_snake(self, id):
        pass

class SnakeHandler(ISnakeHandler):
    def __init__(self):
        self._snakes: Dict[int, ISnake] = {}
        self._dead_snakes = set()

    def kill_snake(self, id):
        return self._dead_snakes.add(id)

    def get_decision(self, id, env_data: EnvData) -> Coord:
        return self._snakes[id].update(env_data)

    def add_snake(self, snake: ISnake):
        self._snakes[snake.get_id()] = snake

    def get_update_order(self) -> list:
        return random.shuffle([id for id in self.snakes.keys() if id not in self._dead_snakes])
