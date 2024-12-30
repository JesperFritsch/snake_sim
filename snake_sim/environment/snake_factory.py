from typing import Dict
import json

from importlib import resources
from snake_sim.utils import DotDict
from snake_sim.snakes.auto_snake import AutoSnake
from snake_sim.snakes.manual_snake import ManualSnake
from snake_sim.snakes.remote_snake import RemoteSnake

with resources.open_text('snake_sim.config', 'default_config.json') as config_file:
    default_config = DotDict(json.load(config_file))

class SnakeFactory:
    def __init__(self):
        self.snake_types = {
            'auto': AutoSnake,
            'manual': ManualSnake,
            'remote': RemoteSnake  # Added RemoteSnake
        }
        self.used_ids = set()

    def create_snake(self, snake_type, **kwargs):
        return self.snake_types[snake_type](**kwargs)

    def create_snakes(self, type_count: Dict[str, int], **kwargs):
        snakes = []
        snake_id = max(self.used_ids) + 1 if self.used_ids else 0
        for snake_type, count in type_count.items():
            for _ in range(count):
                snakes.append(self.create_snake(snake_type, id=snake_id, **kwargs))
                self.used_ids.add(snake_id)
                snake_id += 1
        return snakes