import random
import string
import os
import shutil
import json
import platform
import math
from time import time
from collections.abc import Iterable

from collections import deque

class DotDict(dict):
    def __init__(self, other_dict={}):
        for k, v in other_dict.items():
            if isinstance(v, dict):
                v = DotDict(v)
            # elif isinstance(v, Iterable) and not isinstance(v, str):
            #     v = [DotDict(e) if isinstance(e, dict) else e for e in v]
            self[k] = v

    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError:
            raise AttributeError

    def __setattr__(self, attr, value):
        self[attr] = value

    def __delattr__(self, attr):
        try:
            del self[attr]
        except KeyError:
            raise AttributeError
        
    def update(self, other_dict):
        for k, v in other_dict.items():
            if isinstance(v, dict):
                v = DotDict(v)
            self[k] = v


def exec_time(func):
    def wrapper(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        print(f"Execution time for {func.__name__}: {round((time() - start) * 1000, 3)} ms")
        return result
    return wrapper


def coord_cmp(coord1, coord2):
    return coord1[0] == coord2[0] and coord1[1] == coord2[1]


def distance(c1, c2):
    return math.sqrt(math.pow(c1[0] - c2[0], 2) + math.pow(c1[1] - c2[1], 2))


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
