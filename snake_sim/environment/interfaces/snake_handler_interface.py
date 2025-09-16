from abc import ABC, abstractmethod
from typing import Dict, List

from snake_sim.environment.snake_env import EnvData
from snake_sim.environment.types import Coord
from snake_sim.environment.interfaces.snake_interface import ISnake


class ISnakeHandler(ABC):

    @abstractmethod
    def get_decisions(self, batch_data: Dict[int, EnvData]) -> Dict[int, Coord]:
        pass

    @abstractmethod
    def add_snake(self, snake: ISnake):
        pass

    @abstractmethod
    def get_update_order(self) -> List[int]:
        pass

    @abstractmethod
    def get_batch_order(self, snake_head_positions: Dict[int, Coord]) -> List[List[int]]:
        pass

    @abstractmethod
    def kill_snake(self, id):
        pass

    @abstractmethod
    def get_snakes(self) -> Dict[int, ISnake]:
        pass

    @abstractmethod
    def finalize(self):
        pass