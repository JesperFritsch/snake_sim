from typing import Dict, Tuple
import json

from importlib import resources
from snake_sim.utils import DotDict, SingletonMeta
from snake_sim.snakes.snake import ISnake
from snake_sim.snakes.auto_snake import AutoSnake
from snake_sim.snakes.manual_snake import ManualSnake
from snake_sim.snakes.remote_snake import RemoteSnake

with resources.open_text('snake_sim.config', 'default_config.json') as config_file:
    default_config = DotDict(json.load(config_file))

class SnakeFactory(metaclass=SingletonMeta):
    def __init__(self):
        self.snake_types = {
            'auto': AutoSnake,
            'manual': ManualSnake,
            'remote': RemoteSnake  # Added RemoteSnake
        }
        self.reserved_ids = set()
        self.created_ids = set()

    def get_next_id(self):
        id = max(self.reserved_ids) + 1 if self.reserved_ids else 0
        self.reserved_ids.add(id)
        return id

    def create_snake(self, snake_type, id=None, **kwargs) -> Tuple[int, ISnake]:
        if id is not None:
            if id in self.created_ids:
                raise ValueError(f'ID {id} already in use')
        else:
            id = self.get_next_id()
        self.created_ids.add(id)
        return id, self.snake_types[snake_type](**kwargs)

    def create_snakes(self, type_count: Dict[str, int], **kwargs) -> Dict[int, ISnake]:
        snakes = {}
        for snake_type, count in type_count.items():
            for _ in range(count):
                id, snake = self.create_snake(snake_type, **kwargs)
                snakes[id] = snake
        return snakes