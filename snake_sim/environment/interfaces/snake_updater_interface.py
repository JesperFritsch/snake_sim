from abc import ABC, abstractmethod
from typing import Tuple, List, Dict

from snake_sim.environment.types import Coord, EnvData
from snake_sim.environment.interfaces.snake_interface import ISnake

class ISnakeUpdater(ABC):

    @abstractmethod
    def get_decisions(self, snakes: List[ISnake], env_data: EnvData, timeout: int) -> Dict[int, Coord]: # -> dict of snake id to direction
        pass

    def close(self):
        pass

    def register_snake(self, snake: ISnake):
        pass

    def finalize(self):
        pass

