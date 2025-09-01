

from snake_sim.environment.snake_strategy_factory import SnakeStrategyFactory
from snake_sim.environment.interfaces.snake_strategy_user_interface import ISnakeStrategyUser
from snake_sim.environment.types import SnakeConfig

def apply_strategies(snake: ISnakeStrategyUser, config: SnakeConfig):
    if isinstance(snake, ISnakeStrategyUser) and hasattr(config, "strategies"):
        strategy_factory = SnakeStrategyFactory()
        strategies = strategy_factory.create_strategies(config)
        for prio, strat in strategies.items():
            snake.set_strategy(prio, strat)