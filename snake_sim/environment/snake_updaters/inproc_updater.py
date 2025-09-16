
from typing import Tuple, List, Dict

from snake_sim.environment.interfaces.ISnakeUpdater import ISnakeUpdater
from snake_sim.environment.interfaces.snake_interface import ISnake
from snake_sim.environment.types import Coord, EnvData

class InprocUpdater(ISnakeUpdater):
    def __init__(self):
        pass

    def get_decisions(self, snakes: List[ISnake], env_data: EnvData, timeout: int) -> Dict[int, Coord]:
        decisions = {}
        for snake in snakes:
            decisions[snake.get_id()] = snake.update(env_data)
        return decisions