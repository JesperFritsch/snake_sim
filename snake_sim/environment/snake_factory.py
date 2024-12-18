from typing import Dict
import json

from importlib import resources
from snake_sim.utils import DotDict
from snake_sim.snakes.auto_snake import AutoSnake
from snake_sim.snakes.manual_snake import ManualSnake

with resources.open_text('snake_sim.config', 'default_config.json') as config_file:
    default_config = DotDict(json.load(config_file))

class SnakeFactory:
    def __init__(self):
        self.snake_types = {
            'auto': AutoSnake,
            'manual': ManualSnake
        }

    def create_snake(self, snake_type, **kwargs):
        return self.snake_types[snake_type](**kwargs)

    def create_snakes(self, type_count: Dict[str, int], **kwargs):
        snakes = []
        snake_id = 10
        for snake_type, count in type_count.items():
            while count > 0:
                count -= 1
                snakes.append(self.create_snake(snake_type, id=snake_id, **kwargs))
                snake_id += 2
        return snakes