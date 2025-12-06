
import json
import colorsys
from typing import Dict, Tuple
from snake_sim.environment.types import DotDict

from importlib import resources

with resources.open_text('snake_sim.config', 'default_config.json') as config_file:
    default_config = json.load(config_file)


def generate_distinct_colors(n, s=0.9, v=0.9):
    """
    Generate `n` distinct RGB colors as (r, g, b) tuples in [0, 255],
    all bright and saturated enough to stand out on black.
    """
    colors = []
    for i in range(n):
        h = i / n  # evenly spaced hue
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        colors.append((
            int(r * 255),
            int(g * 255),
            int(b * 255),
        ))
    return colors


def create_color_map(snake_values: dict) -> Dict[int, Tuple[int, int, int]]:
    """ snake_values is a dictionary with snake id as key and a dictionary with 'head_value' and 'body_value' as value """
    config = DotDict(default_config)
    color_map = {config[key]: value for key, value in config.color_mapping.items()}
    nr_snakes = len(snake_values)
    head_colors = generate_distinct_colors(nr_snakes, s=0.7, v=1)
    body_colors = generate_distinct_colors(nr_snakes, s=1, v=0.6)
    for i, snake_value_dict in enumerate(snake_values.values()):
        color_map[snake_value_dict["head_value"]] = head_colors[i]
        color_map[snake_value_dict["body_value"]] = body_colors[i]
    return color_map


def print_colors(colors: list[tuple[int, int, int]]):
    """ Print the given list of RGB colors in the terminal """
    for color in colors:
        r, g, b = color
        print(f"\033[48;2;{r};{g};{b}m   \033[0m", end=' ')
    print()


