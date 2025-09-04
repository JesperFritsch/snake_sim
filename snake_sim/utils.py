import random
import string
import os
import platform
import sys
import math
import numpy as np
import json
from importlib import resources
from time import time
from typing import Dict, Tuple, List

import cProfile
import pstats
from io import StringIO

from importlib import resources

from snake_sim.cpp_bindings.utils import get_locations_with_value

from snake_sim.environment.types import DotDict, Coord

with resources.open_text('snake_sim.config', 'default_config.json') as config_file:
    default_config = json.load(config_file)


class SingletonMeta(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

    @classmethod
    def reset(cls):
        cls._instances.clear()



def exec_time(func):
    def wrapper(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        print(f"Execution time for {func.__name__}: {round((time() - start) * 1000, 3)} ms")
        return result
    return wrapper


def coord_cmp(coord1, coord2):
    return coord1[0] == coord2[0] and coord1[1] == coord2[1]


def distance(c1, c2) -> float:
    return math.sqrt(math.pow(c1[0] - c2[0], 2) + math.pow(c1[1] - c2[1], 2))


def get_locations(s_map: np.ndarray, value, width, height) -> List[Coord]:
    return get_locations_with_value(s_map.flatten(), width, height, value)


def get_coord_parity(coord: Coord) -> int:
    return coord.x % 2 == coord.y % 2


def is_headless():
    if platform.system() == "Windows":
        try:
            import ctypes
            user32 = ctypes.windll.user32
            # Check if there are zero monitors
            return user32.GetSystemMetrics(80) == 0  # SM_CMONITORS
        except Exception:
            return True  # Assume headless if detection fails
    else:
        # For Unix-like systems, check DISPLAY variable
        return os.environ.get("DISPLAY") is None


def coord_op(coord_left, coord_right, op):
    # Check the operation and perform it directly
    if op == '+':
        return tuple(l + r for l, r in zip(coord_left, coord_right))
    elif op == '-':
        return tuple(l - r for l, r in zip(coord_left, coord_right))
    elif op == '*':
        return tuple(l * r for l, r in zip(coord_left, coord_right))
    else:
        raise ValueError("Unsupported operation")


def get_map_files_mapping():
    files = list(resources.files('snake_sim.maps.map_images').iterdir())
    mapping = {f.name.split('.')[0]: f for f in files if f.is_file()}
    mapping.pop('__init__')
    return mapping


def rand_str(n):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=n))


def create_color_map(snake_values: dict) -> Dict[int, Tuple[int, int, int]]:
    """ snake_values is a dictionary with snake id as key and a dictionary with 'head_value' and 'body_value' as value """
    config = DotDict(default_config)
    color_map = {config[key]: value for key, value in config.color_mapping.items()}
    color_len = len(config.snake_colors)
    for i, snake_value_dict in enumerate(snake_values.values()):
        color_map[snake_value_dict["head_value"]] = config.snake_colors[i % color_len]["head_color"]
        color_map[snake_value_dict["body_value"]] = config.snake_colors[i % color_len]["body_color"]
    return color_map


def profile(sort_by='cumtime'):
    def decorator(func):
        def wrapper(*args, **kwargs):
            pr = cProfile.Profile()
            pr.enable()
            result = func(*args, **kwargs)
            pr.disable()
            s = StringIO()
            ps = pstats.Stats(pr, stream=s).sort_stats(sort_by)
            ps.print_stats()
            print(s.getvalue())
            return result
        return wrapper
    return decorator


def print_map(s_map: np.ndarray, free_value: int, food_value: int, blocked_value: int, head_value: int, body_value: int):
    width, height = s_map.shape
    max_nr_digits_width = len(str(width))
    max_nr_digits_height = len(str(height))
    w_nr_strings = [str(i).rjust(max_nr_digits_width) for i in range(height)]
    h_nr_strings = [str(i).rjust(max_nr_digits_height) for i in range(width)]
    digit_rows = [' '.join([f"{nr_string[i]}" for nr_string in w_nr_strings]) for i in range(max_nr_digits_width)]
    map_rows = []
    for i, row in enumerate(s_map):
        map_row = [h_nr_strings[i]]
        for c in row:
            if c == free_value:
                map_row.append('.')
            elif c == food_value:
                map_row.append('F')
            elif c == blocked_value:
                map_row.append('#')
            elif c == head_value:
                map_row.append(f'A')
            elif c == body_value:
                map_row.append('a')
            elif c % 2 == 0:
                map_row.append(f'X')
            else:
                map_row.append(f'x')
        map_row.append(h_nr_strings[i])
        map_rows.append(' '.join(map_row))
    for digit_row in digit_rows:
        print(' ' * (max_nr_digits_height + 1) + digit_row)
    for row in map_rows:
        print(row)
    for digit_row in digit_rows:
        print(' ' * (max_nr_digits_height + 1) + digit_row)
    sys.stdout.flush()