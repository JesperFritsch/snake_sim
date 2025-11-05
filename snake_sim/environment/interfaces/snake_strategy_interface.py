from __future__ import annotations
from abc import ABC, abstractmethod

from snake_sim.environment.types import Coord, StrategyConfig
from snake_sim.environment.interfaces.strategy_snake_interface import IStrategySnake



class ISnakeStrategy(ABC):

    def __init__(self, config: StrategyConfig):
        self._snake: IStrategySnake = None
        self._config = config

    def set_snake(self, snake: IStrategySnake):
        self._snake = snake

    def initialize(self):
        pass 

    @abstractmethod
    def get_wanted_tile() -> Coord:
        pass