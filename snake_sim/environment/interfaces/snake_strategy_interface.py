from abc import ABC, abstractmethod

from snake_sim.environment.types import Coord
from snake_sim.environment.interfaces.snake_interface import ISnake


class ISnakeStrategy(ABC):

    @abstractmethod
    def get_wanted_tile(snake: ISnake) -> Coord:
        pass