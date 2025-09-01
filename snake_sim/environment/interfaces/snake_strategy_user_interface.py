from abc import ABC, abstractmethod

from snake_sim.environment.interfaces.snake_strategy_interface import ISnakeStrategy
from snake_sim.environment.types import Coord


class ISnakeStrategyUser(ABC):

    @abstractmethod
    def set_strategy(self, prio: int, strategy: ISnakeStrategy):
        pass

    @abstractmethod
    def _get_strategy_tile(self) -> Coord:
        pass
