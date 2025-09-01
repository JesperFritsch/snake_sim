from __future__ import annotations
from abc import ABC, abstractmethod

from snake_sim.environment.types import Coord
from snake_sim.environment.interfaces.strategy_snake_interface import IStrategySnake



class ISnakeStrategy(ABC):

    def __init__(self):
        self._snake: IStrategySnake = None

    def set_snake(self, snake: IStrategySnake):
        self._snake = snake

    def initialize(self):
        pass 

    @abstractmethod
    def get_wanted_tile() -> Coord:
        pass