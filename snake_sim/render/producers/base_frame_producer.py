import time
import logging
import numpy as np

from abc import abstractmethod
from pathlib import Path
from typing import Dict, Optional, Tuple

from snake_sim.environment.types import (
    Coord,
    CompleteStepState,
    CurrentIsFirst,
    EnvMetaData,
    LoopStepData,
    NoMoreSteps,
)
from snake_sim.loop_observers.state_builder_observer import StateBuilderObserver
from snake_sim.render.interfaces.frame_producer_interface import IFrameProducer
from snake_sim.render.types import (
    NO_EVENTS,
    Frame,
    FrameEvents,
    FrameInfo,
    Palette,
    SnakeColors,
)
from snake_sim.render.utils import build_color_lut, create_color_map

log = logging.getLogger(Path(__file__).stem)


class BaseFrameProducer(IFrameProducer):
    """ Everything a frame producer needs that is not about drawing.

    Owns the parts that are the same whatever the frames end up looking like:
    waiting for the run to start, mapping frame indices onto simulation steps,
    deciding whether a frame is merely not ready yet or will never exist, keeping
    the palette, indexing what happened on each step, and reporting events over
    the span a consumer actually asked for.

    A subclass supplies two things: `_build`, which prepares whatever it needs
    once the run's metadata arrives and must set `self._size`, and `_draw`, which
    turns one resolved frame into pixels.
    """

    def __init__(
        self,
        state_builder: StateBuilderObserver,
        frames_per_step: int = 1,
        random_colors: bool = False,
    ):
        super().__init__()
        self._state_builder = state_builder
        self._frames_per_step = max(1, int(frames_per_step))
        self._random_colors = random_colors

        self._ready = False
        self._size: Tuple[int, int] = (0, 0)
        self._palette: Optional[Palette] = None
        # value -> RGB, as a dict and as a dense lookup table, for subclasses
        self._color_map: Dict[int, Tuple[int, int, int]] = {}
        self._lut: Optional[np.ndarray] = None
        self._current_frame_idx = 0
        self._last_produced_frame_idx: Optional[int] = None

        # Append-only index of what each snake did, built by walking the steps
        # forward. A memo of a pure function of the steps received so far, so it
        # gives the same answer however the caller arrived at a frame.
        self._eat_steps: Dict[int, list] = {}
        self._ate_on_step: Dict[int, Tuple[int, ...]] = {}
        self._scan_heads: Dict[int, Coord] = {}
        self._steps_scanned = 0

    # ------------------------------------------------------------------ setup

    @abstractmethod
    def _build(self, env: EnvMetaData) -> None:
        """ Prepare for drawing, once the run's metadata is known.

        Must set `self._size` to the (width, height) of the frames it will emit.
        `self._palette`, `self._color_map` and `self._lut` are already filled in.
        """

    def _try_init(self) -> bool:
        if self._ready:
            return True
        start_data = self._state_builder.get_start_data()
        if start_data is None:
            return False
        env = start_data.env_meta_data
        color_map = create_color_map(env.snake_values, rand_colors=self._random_colors)
        lut = build_color_lut(color_map)
        self._color_map = color_map
        self._lut = lut
        self._palette = Palette(
            free=tuple(int(c) for c in lut[env.free_value]),
            food=tuple(int(c) for c in lut[env.food_value]),
            blocked=tuple(int(c) for c in lut[env.blocked_value]),
            snakes={
                s_id: SnakeColors(
                    head=tuple(int(c) for c in lut[values["head_value"]]),
                    body=tuple(int(c) for c in lut[values["body_value"]]),
                )
                for s_id, values in env.snake_values.items()
            },
        )
        self._eat_steps = {s_id: [] for s_id in env.snake_values}
        self._ate_on_step = {}
        self._scan_heads = dict(env.start_positions)
        self._steps_scanned = 0

        self._build(env)
        self._ready = True
        log.debug("%s ready; size=%sx%s frames_per_step=%s",
                  type(self).__name__, self._size[0], self._size[1], self._frames_per_step)
        return True

    def is_ready(self) -> bool:
        return self._try_init()

    def wait_until_ready(self, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout
        while not self._try_init():
            if deadline is not None and time.monotonic() >= deadline:
                return False
            time.sleep(0.005)
        return True

    # --------------------------------------------------------------- timeline

    @property
    def size(self) -> Tuple[int, int]:
        self._try_init()
        return self._size

    @property
    def frames_per_step(self) -> int:
        return self._frames_per_step

    @property
    def palette(self) -> Palette:
        self._try_init()
        return self._palette

    def get_max_step_idx(self) -> int:
        return self._state_builder.get_step_count()

    def get_max_frame_idx(self) -> int:
        return self.get_max_step_idx() * self._frames_per_step

    def get_current_frame_idx(self) -> int:
        return self._current_frame_idx

    def get_current_step_idx(self) -> int:
        return self._current_frame_idx // self._frames_per_step

    # ----------------------------------------------------------------- frames

    @abstractmethod
    def _draw(
        self,
        state: CompleteStepState,
        step_data: Optional[LoopStepData],
        step_idx: int,
        sub_step: int,
    ) -> np.ndarray:
        """ Draw one resolved frame as a (height, width, 3) uint8 RGB buffer. """

    def get_frame_for_step(self, step_idx: int) -> Frame:
        return self.get_frame(step_idx * self._frames_per_step)

    def get_frame(self, frame_idx: int) -> Frame:
        state, step_data, step_idx, sub_step = self._resolve(frame_idx)
        pixels = self._draw(state, step_data, step_idx, sub_step)
        info = FrameInfo(
            frame_idx=frame_idx,
            step_idx=step_idx,
            step_progress=sub_step / self._frames_per_step,
            events=self._events_since_last_frame(frame_idx),
            palette=self._palette,
        )
        return Frame(pixels=pixels, info=info)

    def _resolve(self, frame_idx: int):
        """ Locate a frame in the run and gather what is needed to draw it. """
        if frame_idx < 0:
            raise CurrentIsFirst("Asked for a frame before the start of the run")
        if not self._try_init():
            raise NoMoreSteps("Start data has not been received yet")

        step_idx, sub_step = divmod(frame_idx, self._frames_per_step)
        state = self._state_builder.peek_state(step_idx)

        step_data = None
        if sub_step:
            # Drawing within a step needs the decisions that step made.
            if step_idx >= self._state_builder.get_step_count():
                if self._state_builder.get_stop_data() is not None:
                    raise StopIteration("No more frames available")
                raise NoMoreSteps("Need to receive more steps to produce this frame")
            step_data = self._state_builder.get_step_data(step_idx)

        self._current_frame_idx = frame_idx
        return state, step_data, step_idx, sub_step

    # ----------------------------------------------------------------- events

    def _scan_steps_through(self, step_idx: int):
        """ Extend the run index so every step before `step_idx` has been seen.

        Walks the decisions forward from the start positions, which is enough to
        know where each head landed without building any state.
        """
        limit = min(step_idx, self._state_builder.get_step_count())
        while self._steps_scanned < limit:
            step_data = self._state_builder.get_step_data(self._steps_scanned)
            removed = step_data.removed_food
            ate = []
            for s_id, decision in step_data.decisions.items():
                head = self._scan_heads[s_id] + decision
                self._scan_heads[s_id] = head
                if removed and head in removed:
                    self._eat_steps[s_id].append(self._steps_scanned)
                    ate.append(s_id)
            if ate:
                self._ate_on_step[self._steps_scanned] = tuple(ate)
            self._steps_scanned += 1

    def _events_on(self, frame_idx: int) -> FrameEvents:
        """ The outcome of the step this one frame completes. """
        step_idx, sub_step = divmod(frame_idx, self._frames_per_step)
        if sub_step or step_idx <= 0 or step_idx > self._state_builder.get_step_count():
            return NO_EVENTS
        self._scan_steps_through(step_idx)
        completed = self._state_builder.get_step_data(step_idx - 1)
        ate = self._ate_on_step.get(step_idx - 1, ())
        died = tuple(s_id for s_id, alive in completed.alive_states.items() if not alive)
        if not ate and not died:
            return NO_EVENTS
        return FrameEvents(ate=ate, died=died)

    def _events_since_last_frame(self, frame_idx: int) -> FrameEvents:
        """ Everything that happened between the last frame produced and this one.

        A consumer generally does not ask for every frame - a video at 30 fps
        showing 10 simulation steps per second steps over most of them - so
        reporting only what landed on this exact frame would silently lose the
        steps in between. Reporting the span instead means a consumer gets each
        event once by just reading the frames it actually asked for.
        """
        previous = self._last_produced_frame_idx
        self._last_produced_frame_idx = frame_idx
        if previous is None or frame_idx < previous:
            # First frame, or the caller jumped backwards: there is no span to
            # speak of, so report what belongs to this frame alone.
            return self._events_on(frame_idx)
        if frame_idx == previous:
            # Asking for the same frame again covers no new ground. A consumer
            # running slower than one step per frame lands here, and must not be
            # told about the same eat on every repeat.
            return NO_EVENTS

        stride = self._frames_per_step
        ate: list = []
        died: list = []
        for idx in range(((previous // stride) + 1) * stride, frame_idx + 1, stride):
            events = self._events_on(idx)
            ate.extend(s_id for s_id in events.ate if s_id not in ate)
            died.extend(s_id for s_id in events.died if s_id not in died)
        if not ate and not died:
            return NO_EVENTS
        return FrameEvents(ate=tuple(ate), died=tuple(died))
