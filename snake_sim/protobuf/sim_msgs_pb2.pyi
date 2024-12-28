from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Direction(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    UP: _ClassVar[Direction]
    DOWN: _ClassVar[Direction]
    LEFT: _ClassVar[Direction]
    RIGHT: _ClassVar[Direction]

class MessageType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    RUN_DATA: _ClassVar[MessageType]
    STEP_DATA: _ClassVar[MessageType]
    RUN_META_DATA: _ClassVar[MessageType]
    PIXEL_CHANGES: _ClassVar[MessageType]

class RequestType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    PIXEL_CHANGES_REQ: _ClassVar[RequestType]
    STEP_DATA_REQ: _ClassVar[RequestType]
    RUN_META_DATA_REQ: _ClassVar[RequestType]
UP: Direction
DOWN: Direction
LEFT: Direction
RIGHT: Direction
RUN_DATA: MessageType
STEP_DATA: MessageType
RUN_META_DATA: MessageType
PIXEL_CHANGES: MessageType
PIXEL_CHANGES_REQ: RequestType
STEP_DATA_REQ: RequestType
RUN_META_DATA_REQ: RequestType

class Position(_message.Message):
    __slots__ = ("x", "y")
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    x: int
    y: int
    def __init__(self, x: _Optional[int] = ..., y: _Optional[int] = ...) -> None: ...

class RGB(_message.Message):
    __slots__ = ("r", "g", "b")
    R_FIELD_NUMBER: _ClassVar[int]
    G_FIELD_NUMBER: _ClassVar[int]
    B_FIELD_NUMBER: _ClassVar[int]
    r: int
    g: int
    b: int
    def __init__(self, r: _Optional[int] = ..., g: _Optional[int] = ..., b: _Optional[int] = ...) -> None: ...

class SnakeValues(_message.Message):
    __slots__ = ("body_value", "head_value")
    BODY_VALUE_FIELD_NUMBER: _ClassVar[int]
    HEAD_VALUE_FIELD_NUMBER: _ClassVar[int]
    body_value: int
    head_value: int
    def __init__(self, body_value: _Optional[int] = ..., head_value: _Optional[int] = ...) -> None: ...

class SnakeStep(_message.Message):
    __slots__ = ("snake_id", "curr_head", "prev_head", "curr_tail", "head_dir", "did_eat", "did_turn", "body")
    SNAKE_ID_FIELD_NUMBER: _ClassVar[int]
    CURR_HEAD_FIELD_NUMBER: _ClassVar[int]
    PREV_HEAD_FIELD_NUMBER: _ClassVar[int]
    CURR_TAIL_FIELD_NUMBER: _ClassVar[int]
    HEAD_DIR_FIELD_NUMBER: _ClassVar[int]
    DID_EAT_FIELD_NUMBER: _ClassVar[int]
    DID_TURN_FIELD_NUMBER: _ClassVar[int]
    BODY_FIELD_NUMBER: _ClassVar[int]
    snake_id: int
    curr_head: Position
    prev_head: Position
    curr_tail: Position
    head_dir: Position
    did_eat: bool
    did_turn: str
    body: _containers.RepeatedCompositeFieldContainer[Position]
    def __init__(self, snake_id: _Optional[int] = ..., curr_head: _Optional[_Union[Position, _Mapping]] = ..., prev_head: _Optional[_Union[Position, _Mapping]] = ..., curr_tail: _Optional[_Union[Position, _Mapping]] = ..., head_dir: _Optional[_Union[Position, _Mapping]] = ..., did_eat: bool = ..., did_turn: _Optional[str] = ..., body: _Optional[_Iterable[_Union[Position, _Mapping]]] = ...) -> None: ...

class StepData(_message.Message):
    __slots__ = ("snakes", "food", "step", "full_state")
    SNAKES_FIELD_NUMBER: _ClassVar[int]
    FOOD_FIELD_NUMBER: _ClassVar[int]
    STEP_FIELD_NUMBER: _ClassVar[int]
    FULL_STATE_FIELD_NUMBER: _ClassVar[int]
    snakes: _containers.RepeatedCompositeFieldContainer[SnakeStep]
    food: _containers.RepeatedCompositeFieldContainer[Position]
    step: int
    full_state: bool
    def __init__(self, snakes: _Optional[_Iterable[_Union[SnakeStep, _Mapping]]] = ..., food: _Optional[_Iterable[_Union[Position, _Mapping]]] = ..., step: _Optional[int] = ..., full_state: bool = ...) -> None: ...

class RunMetaData(_message.Message):
    __slots__ = ("width", "height", "food_value", "free_value", "blocked_value", "color_mapping", "snake_ids", "base_map", "snake_values")
    class ColorMappingEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: int
        value: RGB
        def __init__(self, key: _Optional[int] = ..., value: _Optional[_Union[RGB, _Mapping]] = ...) -> None: ...
    class SnakeValuesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: int
        value: SnakeValues
        def __init__(self, key: _Optional[int] = ..., value: _Optional[_Union[SnakeValues, _Mapping]] = ...) -> None: ...
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    FOOD_VALUE_FIELD_NUMBER: _ClassVar[int]
    FREE_VALUE_FIELD_NUMBER: _ClassVar[int]
    BLOCKED_VALUE_FIELD_NUMBER: _ClassVar[int]
    COLOR_MAPPING_FIELD_NUMBER: _ClassVar[int]
    SNAKE_IDS_FIELD_NUMBER: _ClassVar[int]
    BASE_MAP_FIELD_NUMBER: _ClassVar[int]
    SNAKE_VALUES_FIELD_NUMBER: _ClassVar[int]
    width: int
    height: int
    food_value: int
    free_value: int
    blocked_value: int
    color_mapping: _containers.MessageMap[int, RGB]
    snake_ids: _containers.RepeatedScalarFieldContainer[int]
    base_map: _containers.RepeatedScalarFieldContainer[int]
    snake_values: _containers.MessageMap[int, SnakeValues]
    def __init__(self, width: _Optional[int] = ..., height: _Optional[int] = ..., food_value: _Optional[int] = ..., free_value: _Optional[int] = ..., blocked_value: _Optional[int] = ..., color_mapping: _Optional[_Mapping[int, RGB]] = ..., snake_ids: _Optional[_Iterable[int]] = ..., base_map: _Optional[_Iterable[int]] = ..., snake_values: _Optional[_Mapping[int, SnakeValues]] = ...) -> None: ...

class RunData(_message.Message):
    __slots__ = ("run_meta_data", "steps")
    class StepsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: int
        value: StepData
        def __init__(self, key: _Optional[int] = ..., value: _Optional[_Union[StepData, _Mapping]] = ...) -> None: ...
    RUN_META_DATA_FIELD_NUMBER: _ClassVar[int]
    STEPS_FIELD_NUMBER: _ClassVar[int]
    run_meta_data: RunMetaData
    steps: _containers.MessageMap[int, StepData]
    def __init__(self, run_meta_data: _Optional[_Union[RunMetaData, _Mapping]] = ..., steps: _Optional[_Mapping[int, StepData]] = ...) -> None: ...

class MsgWrapper(_message.Message):
    __slots__ = ("type", "payload")
    TYPE_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    type: MessageType
    payload: bytes
    def __init__(self, type: _Optional[_Union[MessageType, str]] = ..., payload: _Optional[bytes] = ...) -> None: ...

class PixelChanges(_message.Message):
    __slots__ = ("pixels", "full_state")
    PIXELS_FIELD_NUMBER: _ClassVar[int]
    FULL_STATE_FIELD_NUMBER: _ClassVar[int]
    pixels: _containers.RepeatedCompositeFieldContainer[PixelData]
    full_state: bool
    def __init__(self, pixels: _Optional[_Iterable[_Union[PixelData, _Mapping]]] = ..., full_state: bool = ...) -> None: ...

class PixelData(_message.Message):
    __slots__ = ("coord", "color")
    COORD_FIELD_NUMBER: _ClassVar[int]
    COLOR_FIELD_NUMBER: _ClassVar[int]
    coord: Position
    color: RGB
    def __init__(self, coord: _Optional[_Union[Position, _Mapping]] = ..., color: _Optional[_Union[RGB, _Mapping]] = ...) -> None: ...

class EnvData(_message.Message):
    __slots__ = ("width", "height", "map", "food", "FOOD_VALUE", "FREE_VALUE", "BLOCKED_VALUE")
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    MAP_FIELD_NUMBER: _ClassVar[int]
    FOOD_FIELD_NUMBER: _ClassVar[int]
    FOOD_VALUE_FIELD_NUMBER: _ClassVar[int]
    FREE_VALUE_FIELD_NUMBER: _ClassVar[int]
    BLOCKED_VALUE_FIELD_NUMBER: _ClassVar[int]
    width: int
    height: int
    map: bytes
    food: _containers.RepeatedCompositeFieldContainer[Position]
    FOOD_VALUE: int
    FREE_VALUE: int
    BLOCKED_VALUE: int
    def __init__(self, width: _Optional[int] = ..., height: _Optional[int] = ..., map: _Optional[bytes] = ..., food: _Optional[_Iterable[_Union[Position, _Mapping]]] = ..., FOOD_VALUE: _Optional[int] = ..., FREE_VALUE: _Optional[int] = ..., BLOCKED_VALUE: _Optional[int] = ...) -> None: ...

class SnakeAction(_message.Message):
    __slots__ = ("action",)
    ACTION_FIELD_NUMBER: _ClassVar[int]
    action: Position
    def __init__(self, action: _Optional[_Union[Position, _Mapping]] = ...) -> None: ...

class Request(_message.Message):
    __slots__ = ("type", "payload")
    TYPE_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    type: RequestType
    payload: bytes
    def __init__(self, type: _Optional[_Union[RequestType, str]] = ..., payload: _Optional[bytes] = ...) -> None: ...

class PixelChangesRequest(_message.Message):
    __slots__ = ("start_step", "end_step")
    START_STEP_FIELD_NUMBER: _ClassVar[int]
    END_STEP_FIELD_NUMBER: _ClassVar[int]
    start_step: int
    end_step: int
    def __init__(self, start_step: _Optional[int] = ..., end_step: _Optional[int] = ...) -> None: ...

class StepDataRequest(_message.Message):
    __slots__ = ("start_step", "end_step", "full_state")
    START_STEP_FIELD_NUMBER: _ClassVar[int]
    END_STEP_FIELD_NUMBER: _ClassVar[int]
    FULL_STATE_FIELD_NUMBER: _ClassVar[int]
    start_step: int
    end_step: int
    full_state: bool
    def __init__(self, start_step: _Optional[int] = ..., end_step: _Optional[int] = ..., full_state: bool = ...) -> None: ...

class RunMetaDataRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class RequestAck(_message.Message):
    __slots__ = ("type", "payload")
    TYPE_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    type: RequestType
    payload: bytes
    def __init__(self, type: _Optional[_Union[RequestType, str]] = ..., payload: _Optional[bytes] = ...) -> None: ...

class RunUpdate(_message.Message):
    __slots__ = ("final_step",)
    FINAL_STEP_FIELD_NUMBER: _ClassVar[int]
    final_step: int
    def __init__(self, final_step: _Optional[int] = ...) -> None: ...
