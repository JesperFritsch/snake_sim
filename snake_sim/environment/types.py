import math
from enum import Enum
from dataclasses import dataclass, field
from collections import deque
from typing import Optional, List, Dict, Deque, Set
import numpy as np


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
    x: int
    y: int
    def distance(self, other):
        return math.sqrt(math.pow(self.x - other[0], 2) + math.pow(self.y - other[1], 2))
    
    def manhattan_distance(self, other):
        return abs(self.x - other[0]) + abs(self.y - other[1])

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

    def __iter__(self):
        yield self.x
        yield self.y

    def __hash__(self):
        return hash((self.x, self.y))

    @property
    def x(self) -> int:
        return self[0]

    @property
    def y(self) -> int:
        return self[1]

    def __repr__(self):
        return f"Coord(x={self[0]}, y={self[1]})"

    def __str__(self):
        return repr(self)

    def __format__(self, format_spec):
        return str(self).__format__(format_spec)


class EnvData:
    def __init__(self, map: bytes, snakes: dict, food_locations: Optional[List[Coord]]):
        self.map = map
        self.snakes = snakes
        self.food_locations = food_locations



class EnvInitData:
    def __init__(self,
                height: int,
                width: int,
                free_value: int,
                blocked_value: int,
                food_value: int,
                snake_values: Dict[int, Dict[str, int]],
                start_positions: Dict[int, Coord],
                base_map: np.ndarray):
        self.height = height
        self.width = width
        self.free_value = free_value
        self.blocked_value = blocked_value
        self.food_value = food_value
        self.snake_values = snake_values
        self.start_positions = start_positions
        self.base_map = base_map


@dataclass
class LoopStartData:
    env_init_data: EnvInitData


@dataclass
class LoopStepData:
    # decisions, snake_grew and snake_times will only have values for alive snakes
    step: int
    total_time: Optional[float] = field(default_factory=float)
    snake_times: Optional[Dict[int, float]] = field(default_factory=dict)
    decisions: Optional[Dict[int, Coord]] = field(default_factory=dict)
    tail_directions: Optional[Dict[int, Coord]] = field(default_factory=dict)
    snake_grew: Optional[Dict[int, bool]] = field(default_factory=dict)
    lengths: Optional[Dict[int, int]] = field(default_factory=dict)
    new_food: Optional[List[Coord]] = field(default_factory=list)
    removed_food: Optional[List[Coord]] = field(default_factory=list)


@dataclass
class LoopStopData:
    pass


@dataclass
class LoopStepState:
    # contains everything needed to recustruct the grid
    food: Set[Coord]
    # heads are at index 0 in the deques
    snake_bodies: Dict[int, Deque]
    

@dataclass
class StrategyConfig:
    type: str
    params: dict = field(default_factory=dict)


@dataclass
class SnakeConfig:
    type: str
    # strategies is a dict of priority (int) -> StrategyConfig
    strategies: Dict[int, StrategyConfig] = field(default_factory=dict)


class SnakeProcType(Enum):
    SHM = 'shm' # Running in a separate process on the same machine, communicating via shared memory
    GRPC = 'grpc' # Running in a separate process or machine, communicating via gRPC