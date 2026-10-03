import numpy as np

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Dict, Mapping, Tuple

RGB = Tuple[int, int, int]


@dataclass(frozen=True)
class SnakeColors:
    head: RGB
    body: RGB


@dataclass(frozen=True)
class Palette:
    """ The colors a producer paints with, for consumers that need to agree with it. """
    free: RGB
    food: RGB
    blocked: RGB
    snakes: Dict[int, SnakeColors] = field(default_factory=dict)


@dataclass(frozen=True)
class FrameEvents:
    """ What happened on a frame, by snake id.

    Events belong to the frame they land on rather than to a stream, so they are
    a pure function of the frame index: seeking, replaying or scrubbing backwards
    reports exactly the same events for the same frame and never double-fires.
    A step's events land on the frame where that step completes, which for a
    producer that interpolates is the frame where the head actually arrives.
    """
    ate: Tuple[int, ...] = ()
    died: Tuple[int, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.ate or self.died)


NO_EVENTS = FrameEvents()


@dataclass(frozen=True)
class FrameInfo:
    """ What a consumer can know about a frame without knowing how it was drawn. """
    frame_idx: int
    step_idx: int
    # 0.0 on the frame a step lands on, rising towards 1.0 across the frames
    # a producer spends animating the way to the next step.
    step_progress: float
    events: FrameEvents
    palette: Palette


@dataclass(frozen=True)
class Frame:
    """ A finished frame: the pixels, plus what they mean. """
    pixels: np.ndarray
    info: FrameInfo


@dataclass(frozen=True)
class GlowConfig:
    """ A pulse of light that runs down a snake's body each time it eats.

    The pulse travels away from the head towards the tail. Its front - the end
    furthest down the body - is the brightest point, slightly lighter than the
    head, and it tapers off back towards the head.
    """
    # How long the pulse is, in cells. 0 or less disables the effect.
    length: int = 4
    # How far the pulse travels down the body for each step the head takes, in cells.
    speed: float = 2.0
    # How far the front is blended towards white, past the snake's head colour.
    # 0 leaves it at the head colour, 1 makes it white.
    brightness: float = 0.35
    # Taper exponent from the front back towards the head. 1 is linear, higher
    # concentrates the light at the front, lower spreads it over the whole pulse.
    falloff: float = 1.5

    @property
    def enabled(self) -> bool:
        return self.length > 0 and self.speed > 0


@dataclass(frozen=True)
class CellScheme:
    """ The raw cell values behind a grid rendering, for character based output. """
    free: int
    food: int
    blocked: int
    # Read-only: consumers share this with the producer that handed it over.
    color_map: Mapping[int, RGB] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self):
        if not isinstance(self.color_map, MappingProxyType):
            object.__setattr__(self, "color_map", MappingProxyType(dict(self.color_map)))
