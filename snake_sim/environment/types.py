import math
from enum import Enum
import numpy as np
from dataclasses import dataclass, field
from collections import deque
from collections.abc import Iterable
from typing import Optional, List, Dict, Deque, Set, Any, Annotated
from pydantic_core import core_schema
from pydantic import (
    BaseModel, ConfigDict, Field,
    BeforeValidator, PlainSerializer, model_validator,
    GetCoreSchemaHandler
)

NdArray = Annotated[
    np.ndarray,
    BeforeValidator(lambda v: v if isinstance(v, np.ndarray) else np.array(v)),
    PlainSerializer(lambda a: a.tolist(), return_type=list),
]
NpDtype = Annotated[
    np.dtype,
    BeforeValidator(lambda v: np.dtype(v)),         # accepts "uint8" or np.dtype
    PlainSerializer(lambda d: str(d), return_type=str),
]


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

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        return core_schema.no_info_after_validator_function(
            cls._coerce,                                   # build the Coord
            core_schema.list_schema(                       # accept [int, int]
                core_schema.int_schema(), min_length=2, max_length=2,
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda c: [c[0], c[1]],                    # emit [x, y]
            ),
        )

    @classmethod
    def _coerce(cls, value):
        return value if isinstance(value, cls) else cls(*value)

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
    margin_frac: float = field(init=False)
    
    def __post_init__(self):
        self.margin_frac = self.margin / self.tile_count if self.tile_count > 0 else 0.0


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


class EnvStepData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    map: NdArray
    snakes: dict[int, dict[str, Any]] # 'is_alive': bool, 'length': int
    food_locations: Optional[List[Coord]]



class EnvMetaData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    height: int
    width: int
    free_value: int
    blocked_value: int
    food_value: int
    snake_tags: dict[int, str]
    snake_values: dict[int, dict[str, int]]
    start_positions: dict[int, Coord]
    base_map: NdArray
    base_map_dtype: NpDtype = Field(default_factory=lambda: np.dtype(np.uint8))

    @model_validator(mode="after")
    def _apply_dtype(self):
        # base_map came back from JSON as a list → np.array infers int64;
        # cast it to the declared dtype.
        if self.base_map.dtype != self.base_map_dtype:
            self.base_map = self.base_map.astype(self.base_map_dtype)
        return self

    def to_dict(self):
        return self.model_dump()

    @classmethod
    def from_dict(cls, d: dict) -> "EnvMetaData":
        return cls.model_validate(d)


class LoopStartData(BaseModel):
    env_meta_data: EnvMetaData


class LoopStepData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
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


class LoopStopData(BaseModel):
    final_step: int


class LoopDecisionData(BaseModel):
    snake_id: int
    wall_time_ns: int


@dataclass
class CompleteStepState:

    env_meta_data: EnvMetaData
    food: Set[Coord]
    # heads are at index 0 in the deques
    snake_bodies: Dict[int, Deque[Coord]]
    snake_alive: Dict[int, bool]
    snake_ate: Dict[int, bool]
    state_idx: int = field(default=0)

    def to_dict(self):
        state_dict = self.__dict__.copy()
        state_dict['env_meta_data'] = self.env_meta_data.to_dict()
        state_dict['food'] = [(f.x, f.y) for f in self.food]
        state_dict['snake_bodies'] = {k: [tuple([*pos]) for pos in v] for k, v in self.snake_bodies.items()}
        return state_dict

    @classmethod
    def from_dict(cls, state_dict):
        state_dict["env_meta_data"] = EnvMetaData.from_dict(state_dict["env_meta_data"])
        state_dict["food"] = set([Coord(*f) for f in state_dict["food"]])
        state_dict["snake_bodies"] = {int(k): deque([Coord(*pos) for pos in v]) for k, v in state_dict["snake_bodies"].items()}
        state_dict["snake_alive"] = {int(k): v for k, v in state_dict["snake_alive"].items()}
        state_dict["snake_ate"] = {int(k): v for k, v in state_dict["snake_ate"].items()}

        instance = cls(
            **state_dict
        )
        return instance
    
    def copy(self) -> 'CompleteStepState':
        return CompleteStepState(
            env_meta_data=self.env_meta_data,
            food=set(self.food),
            snake_bodies={k: deque(v) for k, v in self.snake_bodies.items()},
            snake_alive=self.snake_alive.copy(),
            snake_ate=self.snake_ate.copy(),
            state_idx=self.state_idx,
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
    tag: str
    args: DotDict = field(default_factory=DotDict)
    # strategies is a dict of priority (int) -> StrategyConfig
    strategies: Dict[int, StrategyConfig] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, config_dict):
        strategies = {
            int(k): StrategyConfig(**v)
            for k, v in config_dict.get('strategies', {}).items()
        }
        return cls(type=config_dict['type'], tag=config_dict['tag'], strategies=strategies, args=DotDict(config_dict.get('args', {})))


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
    ext_conn_timeout: int
    ext_init_timeout: int
    start_length: int
    external_snake_configs: List[SnakeConfig]
    distributed_snakes: bool
    snake_configs: list[SnakeConfig]
    decision_timeout: int


@dataclass
class GameConfig(SimConfig):
    player_count: int
    steps_per_sec: int
    player_snake_configs: list[SnakeConfig]


class NoMoreSteps(Exception):
    pass


class CurrentIsFirst(Exception):
    pass
