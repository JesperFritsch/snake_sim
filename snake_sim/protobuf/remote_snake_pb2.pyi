from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Coord(_message.Message):
    __slots__ = ("x", "y")
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    x: int
    y: int
    def __init__(self, x: _Optional[int] = ..., y: _Optional[int] = ...) -> None: ...

class SnakeId(_message.Message):
    __slots__ = ("id",)
    ID_FIELD_NUMBER: _ClassVar[int]
    id: int
    def __init__(self, id: _Optional[int] = ...) -> None: ...

class StartLength(_message.Message):
    __slots__ = ("length",)
    LENGTH_FIELD_NUMBER: _ClassVar[int]
    length: int
    def __init__(self, length: _Optional[int] = ...) -> None: ...

class StartPosition(_message.Message):
    __slots__ = ("start_position",)
    START_POSITION_FIELD_NUMBER: _ClassVar[int]
    start_position: Coord
    def __init__(self, start_position: _Optional[_Union[Coord, _Mapping]] = ...) -> None: ...

class EnvInitData(_message.Message):
    __slots__ = ("height", "width", "free_value", "blocked_value", "food_value", "snake_values", "start_positions", "base_map")
    class SnakeValuesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: int
        value: SnakeValues
        def __init__(self, key: _Optional[int] = ..., value: _Optional[_Union[SnakeValues, _Mapping]] = ...) -> None: ...
    class StartPositionsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: int
        value: Coord
        def __init__(self, key: _Optional[int] = ..., value: _Optional[_Union[Coord, _Mapping]] = ...) -> None: ...
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    FREE_VALUE_FIELD_NUMBER: _ClassVar[int]
    BLOCKED_VALUE_FIELD_NUMBER: _ClassVar[int]
    FOOD_VALUE_FIELD_NUMBER: _ClassVar[int]
    SNAKE_VALUES_FIELD_NUMBER: _ClassVar[int]
    START_POSITIONS_FIELD_NUMBER: _ClassVar[int]
    BASE_MAP_FIELD_NUMBER: _ClassVar[int]
    height: int
    width: int
    free_value: int
    blocked_value: int
    food_value: int
    snake_values: _containers.MessageMap[int, SnakeValues]
    start_positions: _containers.MessageMap[int, Coord]
    base_map: bytes
    def __init__(self, height: _Optional[int] = ..., width: _Optional[int] = ..., free_value: _Optional[int] = ..., blocked_value: _Optional[int] = ..., food_value: _Optional[int] = ..., snake_values: _Optional[_Mapping[int, SnakeValues]] = ..., start_positions: _Optional[_Mapping[int, Coord]] = ..., base_map: _Optional[bytes] = ...) -> None: ...

class SnakeValues(_message.Message):
    __slots__ = ("head_value", "body_value")
    HEAD_VALUE_FIELD_NUMBER: _ClassVar[int]
    BODY_VALUE_FIELD_NUMBER: _ClassVar[int]
    head_value: int
    body_value: int
    def __init__(self, head_value: _Optional[int] = ..., body_value: _Optional[int] = ...) -> None: ...

class EnvData(_message.Message):
    __slots__ = ("map", "snakes")
    class SnakesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: int
        value: SnakeRep
        def __init__(self, key: _Optional[int] = ..., value: _Optional[_Union[SnakeRep, _Mapping]] = ...) -> None: ...
    MAP_FIELD_NUMBER: _ClassVar[int]
    SNAKES_FIELD_NUMBER: _ClassVar[int]
    map: bytes
    snakes: _containers.MessageMap[int, SnakeRep]
    def __init__(self, map: _Optional[bytes] = ..., snakes: _Optional[_Mapping[int, SnakeRep]] = ...) -> None: ...

class SnakeRep(_message.Message):
    __slots__ = ("is_alive", "length")
    IS_ALIVE_FIELD_NUMBER: _ClassVar[int]
    LENGTH_FIELD_NUMBER: _ClassVar[int]
    is_alive: bool
    length: int
    def __init__(self, is_alive: bool = ..., length: _Optional[int] = ...) -> None: ...

class UpdateResponse(_message.Message):
    __slots__ = ("direction",)
    DIRECTION_FIELD_NUMBER: _ClassVar[int]
    direction: Coord
    def __init__(self, direction: _Optional[_Union[Coord, _Mapping]] = ...) -> None: ...

class Empty(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...
