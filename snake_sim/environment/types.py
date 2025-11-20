import math
from enum import Enum
from dataclasses import dataclass, field
from collections import deque
from collections.abc import Iterable
from typing import Optional, List, Dict, Deque, Set, Any
import numpy as np


class DotDict(dict):
    def __init__(self, other_dict={}, **kwargs):
        super().__init__(**kwargs)
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
        if isinstance(other, Iterable):
            return Coord(self.x + other[0], self.y + other[1])
        elif isinstance(other, int):
            return Coord(self.x + other, self.y + other)
        else:
            raise ValueError(f"Can't add {self} and {other}")

    def __sub__(self, other):
        if isinstance(other, Iterable):
            return Coord(self.x - other[0], self.y - other[1])
        elif isinstance(other, int):
            return Coord(self.x - other, self.y - other)
        else:
            raise ValueError(f"Can't sub {self} and {other}")

    def __mul__(self, other):
        if isinstance(other, Iterable):
            return Coord(self.x * other[0], self.y * other[1])
        elif isinstance(other, int):
            return Coord(self.x * other, self.y * other)
        else:
            raise ValueError(f"Can't mult {self} and {other}")

    def __floordiv__(self, other):
        if isinstance(other, Iterable):
            return Coord(self.x // other[0], self.y // other[1])
        elif isinstance(other, int):
            return Coord(self.x // other, self.y // other)
        else:
            raise ValueError(f"Can't floordiv {self} and {other}")

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


@dataclass 
class AreaCheckResult:
    is_clear: bool
    tile_count: int
    total_steps: int
    food_count: int
    has_tail: bool
    margin: int
    needed_steps: int


@dataclass
class RecurseCheckResult:
    best_margin_fracs_at_depth: dict[Coord, dict[int, float]]  # tile -> depth -> margin_frac

    def from_dict(cls, data: dict) -> 'RecurseCheckResult':
        return cls(
            best_margin_fracs_at_depth={
                Coord(*k): {int(depth): float(margin_frac) for depth, margin_frac in v.items()}
                for k, v in data['best_margin_fracs_at_depth'].items()
            }
        )


@dataclass
class EnvStepData:
    map: np.ndarray
    snakes: dict[int, dict[str, Any]] # 'is_alive': bool, 'length': int
    food_locations: Optional[List[Coord]]


@dataclass
class EnvMetaData:
    height: int
    width: int
    free_value: int
    blocked_value: int
    food_value: int
    snake_values: Dict[int, Dict[str, int]]
    start_positions: Dict[int, Coord]
    base_map: np.ndarray
    base_map_dtype: np.dtype = field(default=np.dtype(np.uint8)) # default to uint8

    def to_dict(self):
        meta_dict = self.__dict__.copy()
        meta_dict['base_map'] = self.base_map.tolist()
        meta_dict['start_positions'] = {k: (v.x, v.y) for k, v in self.start_positions.items()}
        meta_dict['base_map_dtype'] = str(self.base_map_dtype)
        return meta_dict

    @classmethod
    def from_dict(cls, meta_dict):
        dtype = np.dtype(meta_dict['base_map_dtype'])
        return cls(
            height=meta_dict['height'],
            width=meta_dict['width'],
            free_value=meta_dict['free_value'],
            blocked_value=meta_dict['blocked_value'],
            food_value=meta_dict['food_value'],
            snake_values={int(k): v for k, v in meta_dict['snake_values'].items()},
            start_positions={int(k): Coord(*v) for k, v in meta_dict['start_positions'].items()},
            base_map=np.array(meta_dict['base_map'], dtype=dtype),
            base_map_dtype=dtype
        )

@dataclass
class LoopStartData:
    env_meta_data: EnvMetaData


@dataclass
class LoopStepData:
    # decisions, snake_grew and snake_times will only have values for alive snakes
    step: int
    total_time: float
    alive_states: Dict[int, bool]
    snake_times: Dict[int, float]
    decisions: Dict[int, Coord]
    tail_directions: Dict[int, Coord]
    snake_grew: Dict[int, bool]
    lengths: Dict[int, int]
    new_food: List[Coord]
    removed_food: List[Coord]


@dataclass
class LoopStopData:
    final_step: int


@dataclass
class CompleteStepState:

    env_meta_data: EnvMetaData
    food: Set[Coord]
    # heads are at index 0 in the deques
    snake_bodies: Dict[int, Deque[Coord]]
    snake_alive: Dict[int, bool]
    state_idx: int = field(default=0)

    def to_dict(self):
        state_dict = self.__dict__.copy()
        state_dict['env_meta_data'] = self.env_meta_data.to_dict()
        state_dict['food'] = [(f.x, f.y) for f in self.food]
        state_dict['snake_bodies'] = {k: [tuple([*pos]) for pos in v] for k, v in self.snake_bodies.items()}
        return state_dict

    @classmethod
    def from_dict(cls, state_dict):
        instance = cls(
            env_meta_data=EnvMetaData.from_dict(state_dict['env_meta_data']),
            food=set([Coord(*f) for f in state_dict['food']]),
            snake_bodies={int(k): deque([Coord(*pos) for pos in v]) for k, v in state_dict['snake_bodies'].items()},
            snake_alive={int(k): v for k, v in state_dict['snake_alive'].items()},
        )
        instance.state_idx = state_dict['state_idx']
        return instance
    
    def copy(self) -> 'CompleteStepState':
        return CompleteStepState(
            env_meta_data=self.env_meta_data,
            food=set(self.food),
            snake_bodies={k: deque(v) for k, v in self.snake_bodies.items()},
            snake_alive=self.snake_alive.copy(),
            state_idx=self.state_idx
        )

@dataclass
class StrategyConfig:
    type: str
    params: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, config_dict):
        return cls(
            type=config_dict['type'],
            params=config_dict.get('params', {})
        )


@dataclass
class SnakeConfig:
    type: str
    args: DotDict = field(default_factory=DotDict)
    # strategies is a dict of priority (int) -> StrategyConfig
    strategies: Dict[int, StrategyConfig] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, config_dict):
        strategies = {
            int(k): StrategyConfig(**v)
            for k, v in config_dict.get('strategies', {}).items()
        }
        return cls(type=config_dict['type'], strategies=strategies, args=DotDict(config_dict.get('args', {})))


class SnakeProcType(Enum):
    SHM = 'shm' # Running in a separate process on the same machine, communicating via shared memory
    GRPC = 'grpc' # Running in a separate process or machine, communicating via gRPC


@dataclass
class SimConfig:
    map: str
    food: int
    height: int
    width: int
    food_decay: int
    snake_count: int
    calc_timeout: int
    verbose: bool
    start_length: int
    external_snake_targets: List[str]
    inproc_snakes: bool
    snake_config: SnakeConfig


@dataclass
class GameConfig(SimConfig):
    player_count: int
    spm: int
    snake_game_config: SnakeConfig