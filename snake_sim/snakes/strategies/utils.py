

from snake_sim.environment.snake_strategy_factory import SnakeStrategyFactory
from snake_sim.environment.interfaces.strategy_snake_interface import IStrategySnake
from snake_sim.environment.types import SnakeConfig

def apply_strategies(snake: IStrategySnake, config: SnakeConfig):
    if isinstance(snake, IStrategySnake) and hasattr(config, "strategies"):
        strategy_factory = SnakeStrategyFactory()
        strategies = strategy_factory.create_strategies(config)
        for prio, strat in strategies.items():
            snake.set_strategy(prio, strat)