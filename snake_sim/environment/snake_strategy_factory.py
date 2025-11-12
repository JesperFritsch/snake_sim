from typing import Dict

from snake_sim.environment.types import SnakeConfig, StrategyConfig
from snake_sim.snakes.strategies.food_strategy import FoodSeeker
from snake_sim.environment.interfaces.snake_strategy_interface import ISnakeStrategy


class SnakeStrategyFactory:
    _strategy_classes = {
        'food_seeker': FoodSeeker,
    }

    def create_strategy(self, strategy_name: str, strategy_config: StrategyConfig) -> ISnakeStrategy:
        try:
            strat_class = self._strategy_classes[strategy_name]
        except KeyError:
            raise ValueError(f"Unknown strategy type: {strategy_name}")
        # Pass params dict directly into strategy if it accepts config
        return strat_class(strategy_config)

    def create_strategies(self, config: SnakeConfig) -> Dict[int, ISnakeStrategy]:
        strategies_dict = config.strategies
        strategies = {}
        for priority, strat_config in strategies_dict.items():
            strategies[priority] = self.create_strategy(strat_config.type, strat_config.params)
        return strategies