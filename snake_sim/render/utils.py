
import json
import colorsys
import random
from typing import Dict, Tuple
from snake_sim.environment.types import DotDict

from importlib import resources

with resources.open_text('snake_sim.config', 'default_config.json') as config_file:
    default_config = json.load(config_file)

class HSVColor:
    def __init__(self, h: float, s: float = 1, v: float = 1):
        self.h = h
        self.s = s
        self.v = v

    def to_rgb(self) -> Tuple[int, int, int]:
        r, g, b = colorsys.hsv_to_rgb(self.h, self.s, self.v)
        return (
            int(r * 255),
            int(g * 255),
            int(b * 255),
        )
    
    @staticmethod
    def from_rgb(r: int, g: int, b: int) -> 'HSVColor':
        h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
        return HSVColor(h, s, v)
    
    def copy_with(self, h: float | None = None, s: float | None = None, v: float | None = None) -> 'HSVColor':
        return HSVColor(
            max(0, min(1, h if h is not None else self.h)),
            max(0, min(1, s if s is not None else self.s)),
            max(0, min(1, v if v is not None else self.v)),
        )


def create_color_map(snake_values: dict, rand_colors: bool = False) -> Dict[int, Tuple[int, int, int]]:
    """ snake_values is a dictionary with snake id as key and a dictionary with 'head_value' and 'body_value' as value """
    config = DotDict(default_config)
    color_map = {config[key]: value for key, value in config.color_mapping.items()}
    for i, snake_value_dict in enumerate(snake_values.values()):
        if rand_colors:
            color = HSVColor(random.random())
        else:
            color = HSVColor(i / len(snake_values))
        color_map[snake_value_dict["head_value"]] = color.copy_with(s=0.7).to_rgb()
        color_map[snake_value_dict["body_value"]] = color.copy_with(s=1, v=0.6).to_rgb()
    return color_map


def print_colors(colors: list[tuple[int, int, int]]):
    """ Print the given list of RGB colors in the terminal """
    for color in colors:
        r, g, b = color
        print(f"\033[48;2;{r};{g};{b}m   \033[0m", end=' ')
    print()


