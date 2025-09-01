

from snake_sim.environment.types import Coord
from snake_sim.environment.interfaces.snake_strategy_interface import ISnakeStrategy


class FoodSeeker(ISnakeStrategy):
    """ A simple strategy that tries to get to the closest food """
    def get_wanted_tile(self, snake) -> Coord:
        return None