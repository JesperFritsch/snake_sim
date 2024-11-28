from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Empty(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class Position(_message.Message):
    __slots__ = ("x", "y")
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    x: int
    y: int
    def __init__(self, x: _Optional[int] = ..., y: _Optional[int] = ...) -> None: ...

class SnakeAction(_message.Message):
    __slots__ = ("action",)
    ACTION_FIELD_NUMBER: _ClassVar[int]
    action: Position
    def __init__(self, action: _Optional[_Union[Position, _Mapping]] = ...) -> None: ...

class EnvData(_message.Message):
    __slots__ = ("map", "food")
    MAP_FIELD_NUMBER: _ClassVar[int]
    FOOD_FIELD_NUMBER: _ClassVar[int]
    map: _containers.RepeatedScalarFieldContainer[int]
    food: _containers.RepeatedCompositeFieldContainer[Position]
    def __init__(self, map: _Optional[_Iterable[int]] = ..., food: _Optional[_Iterable[_Union[Position, _Mapping]]] = ...) -> None: ...

class InitEnvData(_message.Message):
    __slots__ = ("width", "height", "FOOD_TILE", "FREE_TILE", "BLOCKED_TILE", "env_data")
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    FOOD_TILE_FIELD_NUMBER: _ClassVar[int]
    FREE_TILE_FIELD_NUMBER: _ClassVar[int]
    BLOCKED_TILE_FIELD_NUMBER: _ClassVar[int]
    ENV_DATA_FIELD_NUMBER: _ClassVar[int]
    width: int
    height: int
    FOOD_TILE: int
    FREE_TILE: int
    BLOCKED_TILE: int
    env_data: EnvData
    def __init__(self, width: _Optional[int] = ..., height: _Optional[int] = ..., FOOD_TILE: _Optional[int] = ..., FREE_TILE: _Optional[int] = ..., BLOCKED_TILE: _Optional[int] = ..., env_data: _Optional[_Union[EnvData, _Mapping]] = ...) -> None: ...
