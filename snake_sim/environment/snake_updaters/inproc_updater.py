
from typing import Tuple, List, Dict

from snake_sim.environment.interfaces.snake_updater_interface import ISnakeUpdater
from snake_sim.environment.interfaces.snake_interface import ISnake
from snake_sim.environment.types import Coord, EnvStepData

class InprocUpdater(ISnakeUpdater):
    def __init__(self):
        super().__init__()
        pass

    def get_decisions(self, snakes: List[ISnake], env_step_data: EnvStepData, timeout: float) -> Dict[int, Coord]:
        decisions = {}
        for snake in snakes:
            decisions[snake.get_id()] = snake.update(env_step_data)
        return decisions