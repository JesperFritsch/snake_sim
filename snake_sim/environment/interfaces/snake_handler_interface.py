from abc import ABC, abstractmethod

from snake_sim.environment.snake_env import EnvData
from snake_sim.utils import Coord
from snake_sim.environment.interfaces.snake_interface import ISnake


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

    @abstractmethod
    def get_snakes(self):
        pass