import random
import string
import os
import platform
import math
from time import time

from importlib import resources

class DotDict(dict):
    def __init__(self, other_dict={}):
        for k, v in other_dict.items():
            if isinstance(v, dict):
                v = DotDict(v)
            # elif isinstance(v, Iterable) and not isinstance(v, str):
            #     v = [DotDict(e) if isinstance(e, dict) else e for e in v]
            self[k.lower()] = v

    def __getattr__(self, attr):
        try:
            return self[attr.lower()]
        except KeyError:
            raise AttributeError

    def __setattr__(self, attr, value):
        self[attr.lower()] = value

    def __delattr__(self, attr):
        try:
            del self[attr]
        except KeyError:
            raise AttributeError


class Coord(tuple):
    def __new__(cls, x, y):
        return super(Coord, cls).__new__(cls, (x, y))

    def __reduce__(self):
        return (self.__class__, (self[0], self[1]))

    def __add__(self, other):
        return Coord(self.x + other[0], self.y + other[1])

    def __sub__(self, other):
        return Coord(self.x - other[0], self.y - other[1])

    def __mul__(self, other):
        return Coord(self.x * other[0], self.y * other[1])

    def __eq__(self, other):
        return (
                (isinstance(other, tuple) or isinstance(other, list)) and
                len(self) == len(other) and
                self.x == other[0] and
                self.y == other[1]
            )

    def __hash__(self):
        return hash((self.x, self.y))

    @property
    def x(self):
        return self[0]

    @property
    def y(self):
        return self[1]

    def __repr__(self):
        return f"Coord(x={self[0]}, y={self[1]})"

    def __str__(self):
        return repr(self)
    
    def __format__(self, format_spec):
        return str(self).__format__(format_spec)


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


def get_map_files_mapping():
    files = list(resources.files('snake_sim.maps.map_images').iterdir())
    return {f.name.split('.')[0]: f for f in files if f.is_file()}


def rand_str(n):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=n))
