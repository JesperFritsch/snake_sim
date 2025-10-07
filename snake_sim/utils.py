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

from pathlib import Path
from importlib import resources

from snake_sim.cpp_bindings.utils import get_locations_with_value

from snake_sim.environment.types import DotDict, Coord, CompleteStepState

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

def save_step_state(step_state: CompleteStepState):
    """ Save the step state to a json file. """
    file_path = Path(__file__).parent.parent / "test_bench" / "state_files" / f"step_{rand_str(10)}.json"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = step_state.to_dict()
    with open(file_path, 'w') as f:
        json.dump(state_dict, f, indent=4)