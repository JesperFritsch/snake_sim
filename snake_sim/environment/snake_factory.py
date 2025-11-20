from typing import List
import json

from importlib import resources
from snake_sim.utils import SingletonMeta
from snake_sim.snakes.snake_base import ISnake
from snake_sim.snakes.grpc_proxy_snake import GRPCProxySnake
from snake_sim.snakes.shm_proxy_snake import SHMProxySnake
from snake_sim.snakes.survivor_snake import SurvivorSnake
from snake_sim.rl.snakes.ppo_snake import PPOSnake
from snake_sim.environment.types import DotDict, SnakeConfig, SnakeProcType
from snake_sim.snakes.strategies.utils import apply_strategies


TYPENAME_TO_CLASS = {
    'survivor': SurvivorSnake,
    'ai_ppo': PPOSnake,
}

with resources.open_text('snake_sim.config', 'default_config.json') as config_file:
    default_config = DotDict(json.load(config_file))

class SnakeFactory(metaclass=SingletonMeta):
    """ Factory for creating snakes of different types
    Starts processes for remote snakes if no target is provided.
    """
    def __init__(self):
        pass

    def _create_proxy_snake(self, target: str, proc_type: SnakeProcType) -> ISnake:
        if proc_type == SnakeProcType.GRPC:
            return GRPCProxySnake(target)
        elif proc_type == SnakeProcType.SHM:
            return SHMProxySnake(target)

    def _create_snake_from_config(self, snake_config: SnakeConfig) -> ISnake:
        try:
            snake_class = TYPENAME_TO_CLASS[snake_config.type]
        except KeyError:
            raise ValueError(f"Unknown snake type: {snake_config.type}")
        print(f"Creating snake of type: {snake_config.type}")
        print(f"With args: {snake_config.args}")
        snake = snake_class(**snake_config.args)
            
        apply_strategies(snake, snake_config)
        return snake

    def create_snake(
            self,
            proc_type: SnakeProcType=None,
            snake_config: SnakeConfig=None,
            target: str=None
        ) -> ISnake:
        """
        Creates a snake of the given type. If snake_config is provided,
        creates a local snake based on the config.
        If proc_type and target are provided, creates a proxy snake that connects to the given target.
        """
        if snake_config:
            return self._create_snake_from_config(snake_config)
        elif proc_type and target:
            proxy_snake = self._create_proxy_snake(target, proc_type)
            return proxy_snake
        else:
            raise ValueError("Either snake_config or both proc_type and target must be provided")

    def create_many_snakes(
            self,
            snake_config: SnakeConfig,
            count: int
        ) -> List[ISnake]:
        """ Creates multiple local snakes based on the given configuration. """
        snakes = []
        for _ in range(count):
            snake = self.create_snake(snake_config=snake_config)
            snakes.append(snake)
        return snakes
