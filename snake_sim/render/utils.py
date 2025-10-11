
import json

from typing import Dict, Tuple
from snake_sim.environment.types import DotDict

from importlib import resources

with resources.open_text('snake_sim.config', 'default_config.json') as config_file:
    default_config = json.load(config_file)

def create_color_map(snake_values: dict) -> Dict[int, Tuple[int, int, int]]:
    """ snake_values is a dictionary with snake id as key and a dictionary with 'head_value' and 'body_value' as value """
    config = DotDict(default_config)
    color_map = {config[key]: value for key, value in config.color_mapping.items()}
    color_len = len(config.snake_colors)
    for i, snake_value_dict in enumerate(snake_values.values()):
        color_map[snake_value_dict["head_value"]] = config.snake_colors[i % color_len]["head_color"]
        color_map[snake_value_dict["body_value"]] = config.snake_colors[i % color_len]["body_color"]
    return color_map