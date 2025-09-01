from typing import Dict, Tuple

from snake_sim.environment.types import Coord
from snake_sim.environment.interfaces.snake_strategy_interface import ISnakeStrategy
from snake_sim.environment.interfaces.snake_strategy_user_interface import ISnakeStrategyUser

class StrategySnakeMixin(ISnakeStrategyUser):
    def __init__(self):
        super().__init__()
        self._strategies: Dict[int, ISnakeStrategy] = {}

    def set_strategy(self, prio: int, strategy: ISnakeStrategy):
        self._strategies[prio] = strategy

    def _get_strategy_tile(self) -> Coord:
        for prio in sorted(self._strategies.keys()):
            strat = self._strategies[prio]
            tile = strat.get_wanted_tile(self)
            if tile is not None:
                return tile
        return None