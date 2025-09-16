from typing import Dict, Tuple
import json
from enum import Enum

from importlib import resources
from snake_sim.utils import SingletonMeta
from snake_sim.snakes.snake_base import ISnake
from snake_sim.snakes.grpc_proxy_snake import GRPCProxySnake
from snake_sim.snakes.survivor_snake import SurvivorSnake
from snake_sim.environment.snake_processes import ProcessPool
from snake_sim.environment.types import DotDict, SnakeConfig

class SnakeProcType(Enum):
    LOCAL = 'local' # Running in this process
    REMOTE = 'remote' # Running in a separate process or machine


TYPENAME_TO_CLASS = {
    'survivor': SurvivorSnake,
}

with resources.open_text('snake_sim.config', 'default_config.json') as config_file:
    default_config = DotDict(json.load(config_file))

class SnakeFactory(metaclass=SingletonMeta):
    """ Factory for creating snakes of different types 
    Starts processes for remote snakes if no target is provided.
    """
    def __init__(self):
        self._created_ids = set()

    def _get_next_id(self):
        id = max(self._created_ids) + 1 if self._created_ids else 0
        self._created_ids.add(id)
        return id

    def _create_remote_snake_from_target(self, target: str) -> Tuple[int, ISnake]:
        id = self._get_next_id()
        return id, GRPCProxySnake(target=target)

    def _create_remote_snake_from_config(self, snake_config: SnakeConfig) -> Tuple[int, ISnake]:
        id = self._get_next_id()
        ProcessPool().start(id, snake_config=snake_config)
        target = ProcessPool().get_target(id)
        return id, GRPCProxySnake(target=target)
    
    def _create_inprocess_snake_from_config(self, snake_config: SnakeConfig) -> Tuple[int, ISnake]:
        id = self._get_next_id()
        try:
            snake_class = TYPENAME_TO_CLASS[snake_config.type]
        except KeyError:
            raise ValueError(f"Unknown snake type: {snake_config.type}")
        snake = snake_class()
        return id, snake

    def create_snake(
            self, 
            proc_type: SnakeProcType, 
            snake_config: SnakeConfig=None, 
            target: str=None
        ) -> Tuple[int, ISnake]:
        """ Creates a snake of the given type.
        If proc_type is REMOTE and target is None, a new process will be started for the snake.
        If proc_type is LOCAL, a new instance of the snake will be created in this process.
        If proc_type is REMOTE and target is provided, a GRPCProxySnake will be created that connects to the given target.

        Args:
            proc_type (SnakeProcType): The type of process to create the snake in.
            snake_config (SnakeConfig, optional): Configuration for the snake. Required if proc_type is LOCAL or if proc_type is REMOTE and target is None.
            target (str, optional): The target address for the remote snake. Required if proc_type is REMOTE and no snake_config is provided.

        Returns:
            Tuple[int, ISnake]: The id and instance of the created snake.
        """
        if proc_type == SnakeProcType.LOCAL:
            if not snake_config:
                raise ValueError("snake_config must be provided for LOCAL snakes")
            id, instance = self._create_inprocess_snake_from_config(snake_config)
        elif proc_type == SnakeProcType.REMOTE:
            if target:
                id, instance = self._create_remote_snake_from_target(target)
            elif snake_config:
                id, instance = self._create_remote_snake_from_config(snake_config)
            else:
                raise ValueError("Either target or snake_config must be provided for REMOTE snakes")
        else:
            raise ValueError(f"Unknown SnakeProcType: {proc_type}")
        self._created_ids.add(id)
        return id, instance

    def create_many_snakes(
            self, 
            proc_type: SnakeProcType,
            snake_config: SnakeConfig,
            count: int
        ) -> Dict[int, ISnake]:
        """ Creates multiple snakes of the given type. """
        snakes = {}
        for _ in range(count):
            id, snake = self.create_snake(proc_type, snake_config)
            snakes[id] = snake
        return snakes
