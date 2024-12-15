import random
from typing import Dict

from snake_sim.snakes.snake import ISnake
from snake_sim.environment.snake_env import EnvData
from snake_sim.utils import Coord

from snake_sim.environment.interfaces.snake_handler_interface import ISnakeHandler


class SnakeHandler(ISnakeHandler):
    def __init__(self):
        self._snakes: Dict[int, ISnake] = {}
        self._dead_snakes = set()

    def get_snakes(self):
        return self._snakes.values()

    def kill_snake(self, id):
        return self._dead_snakes.add(id)

    def get_decision(self, id, env_data: EnvData) -> Coord:
        return self._snakes[id].update(dict(env_data))

    def add_snake(self, snake: ISnake):
        self._snakes[snake.get_id()] = snake

    def get_update_order(self) -> list:
        return random.shuffle([id for id in self._snakes.keys() if id not in self._dead_snakes])
