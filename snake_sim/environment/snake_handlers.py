import random
from typing import Dict, List

from snake_sim.snakes.snake import ISnake
from snake_sim.environment.snake_env import EnvData
from snake_sim.utils import Coord

from snake_sim.environment.interfaces.snake_handler_interface import ISnakeHandler


class SnakeHandler(ISnakeHandler):
    def __init__(self):
        self._snakes: Dict[int, ISnake] = {}
        self._dead_snakes = set()

    def get_snakes(self) -> Dict[int, ISnake]:
        return self._snakes.copy()

    def kill_snake(self, id):
        return self._dead_snakes.add(id)

    def get_decision(self, id, env_data: EnvData) -> Coord:
        snake = self._snakes[id]
        decision = snake.update(env_data)
        try:
            decision_coord = Coord(*decision)
        except:
            return
        return decision_coord

    def add_snake(self, id, snake: ISnake):
        self._snakes[id] = snake

    def get_update_order(self) -> list:
        ids = [id for id in self._snakes.keys() if id not in self._dead_snakes]
        random.shuffle(ids)
        return ids
