import time
import logging
import numpy as np

from bisect import bisect_left

from pathlib import Path
from typing import Any, Deque, Dict, Optional, Tuple

from snake_sim.environment.types import (
    Coord,
    CompleteStepState,
    CurrentIsFirst,
    LoopStepData,
    NoMoreSteps,
)
from snake_sim.loop_observers.state_builder_observer import StateBuilderObserver
from snake_sim.map_utils.general import expand_map
from snake_sim.render.interfaces.frame_producer_interface import ICellGridProducer
from snake_sim.render.types import (
    NO_EVENTS,
    CellScheme,
    Frame,
    FrameEvents,
    FrameInfo,
    GlowConfig,
    Palette,
    SnakeColors,
)
from snake_sim.render.utils import build_color_lut, create_color_map

log = logging.getLogger(Path(__file__).stem)


class GridFrameProducer(ICellGridProducer):
    """ Produces the plain one-pixel-per-cell grid rendering.

    Builds every frame from scratch out of the simulation state, so a frame is a
    pure function of (step, sub-step) and seeking anywhere is just as correct as
    playing forward. `expansion` scales the grid up and spends the extra pixels
    on interpolating snake movement: with expansion 4 a snake advances one pixel
    per frame over four frames instead of jumping a whole cell at once.

    Pixels and raw cell values come out of the same compose pass, so the two can
    never disagree about where a snake is.
    """

    def __init__(
        self,
        state_builder: StateBuilderObserver,
        expansion: int = 1,
        random_colors: bool = False,
        glow: GlowConfig | None = None,
    ):
        super().__init__()
        self._state_builder = state_builder
        self._expansion = max(1, int(expansion))
        self._random_colors = random_colors
        self._glow = glow if (glow is not None and glow.enabled) else None
        # Append-only index of the steps each snake ate on, plus how far the scan
        # has got. A memo of a pure function of the received steps, so it gives
        # the same answer however the caller arrived at a frame.
        self._eat_steps: Dict[int, list] = {}
        self._ate_on_step: Dict[int, Tuple[int, ...]] = {}
        self._last_produced_frame_idx: Optional[int] = None
        self._scan_heads: Dict[int, Coord] = {}
        self._steps_scanned = 0
        self._glow_rgb: Dict[int, np.ndarray] = {}
        self._current_frame_idx = 0
        self._ready = False
        self._size: Tuple[int, int] = (0, 0)
        self._palette: Optional[Palette] = None
        self._cell_scheme: Optional[CellScheme] = None
        self._base_rgb: Optional[np.ndarray] = None
        self._base_values: Optional[np.ndarray] = None
        self._food_rgb: Optional[np.ndarray] = None
        self._food_value: int = 0
        self._snake_rgb: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self._snake_values: Dict[int, Tuple[int, int]] = {}

    # ------------------------------------------------------------------ setup

    def _try_init(self) -> bool:
        if self._ready:
            return True
        start_data = self._state_builder.get_start_data()
        if start_data is None:
            return False
        env = start_data.env_meta_data
        color_map = create_color_map(env.snake_values, rand_colors=self._random_colors)
        lut = build_color_lut(color_map)

        self._base_values = expand_map(env.base_map, self._expansion, env.free_value, env.blocked_value)
        self._base_rgb = np.ascontiguousarray(lut[self._base_values.astype(np.intp, copy=False)])

        self._food_value = env.food_value
        self._food_rgb = lut[env.food_value]
        self._snake_values = {
            s_id: (values["head_value"], values["body_value"])
            for s_id, values in env.snake_values.items()
        }
        self._snake_rgb = {
            s_id: (lut[head], lut[body]) for s_id, (head, body) in self._snake_values.items()
        }

        self._palette = Palette(
            free=tuple(int(c) for c in lut[env.free_value]),
            food=tuple(int(c) for c in lut[env.food_value]),
            blocked=tuple(int(c) for c in lut[env.blocked_value]),
            snakes={
                s_id: SnakeColors(
                    head=tuple(int(c) for c in lut[head]),
                    body=tuple(int(c) for c in lut[body]),
                )
                for s_id, (head, body) in self._snake_values.items()
            },
        )
        self._cell_scheme = CellScheme(
            free=env.free_value,
            food=env.food_value,
            blocked=env.blocked_value,
            color_map=color_map,
        )

        if self._glow is not None:
            # The front of the pulse sits just past the head colour, towards white.
            for s_id, (head, _) in self._snake_rgb.items():
                head_f = head.astype(np.float32)
                self._glow_rgb[s_id] = head_f + self._glow.brightness * (255.0 - head_f)
        self._eat_steps = {s_id: [] for s_id in env.snake_values}
        self._ate_on_step = {}
        self._scan_heads = dict(env.start_positions)

        height, width = self._base_rgb.shape[:2]
        self._size = (width, height)
        self._ready = True
        log.debug("GridFrameProducer ready; size=%sx%s expansion=%s", width, height, self._expansion)
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
        return self._expansion

    @property
    def palette(self) -> Palette:
        self._try_init()
        return self._palette

    @property
    def cell_scheme(self) -> CellScheme:
        self._try_init()
        return self._cell_scheme

    def get_max_step_idx(self) -> int:
        return self._state_builder.get_step_count()

    def get_max_frame_idx(self) -> int:
        return self.get_max_step_idx() * self._expansion

    def get_current_frame_idx(self) -> int:
        return self._current_frame_idx

    def get_current_step_idx(self) -> int:
        return self._current_frame_idx // self._expansion

    # ----------------------------------------------------------------- frames

    def get_frame_for_step(self, step_idx: int) -> Frame:
        return self.get_frame(step_idx * self._expansion)

    def get_frame(self, frame_idx: int) -> Frame:
        state, step_data, step_idx, sub_step = self._resolve(frame_idx)
        paths = self._snake_paths(state, step_data, sub_step)
        pixels = self._compose(self._base_rgb, self._food_rgb, self._snake_rgb, state, paths)
        if self._glow is not None:
            self._apply_glow(pixels, paths, step_idx, sub_step)
        info = FrameInfo(
            frame_idx=frame_idx,
            step_idx=step_idx,
            step_progress=sub_step / self._expansion,
            events=self._events_since_last_frame(frame_idx),
            palette=self._palette,
        )
        return Frame(pixels=pixels, info=info)

    def get_cell_grid(self, frame_idx: int) -> np.ndarray:
        # No glow here: a cell value says what a cell *is*, and the glow is a
        # colour on top of that, not a different kind of cell.
        state, step_data, _, sub_step = self._resolve(frame_idx)
        paths = self._snake_paths(state, step_data, sub_step)
        return self._compose(self._base_values, self._food_value, self._snake_values, state, paths)

    # ------------------------------------------------------------------ guts

    def _resolve(self, frame_idx: int):
        """ Locate a frame in the run and gather what is needed to draw it. """
        if frame_idx < 0:
            raise CurrentIsFirst("Asked for a frame before the start of the run")
        if not self._try_init():
            raise NoMoreSteps("Start data has not been received yet")

        step_idx, sub_step = divmod(frame_idx, self._expansion)
        state = self._state_builder.peek_state(step_idx)

        step_data = None
        if sub_step:
            # Interpolating within a step needs the decisions that step made.
            if step_idx >= self._state_builder.get_step_count():
                if self._state_builder.get_stop_data() is not None:
                    raise StopIteration("No more frames available")
                raise NoMoreSteps("Need to receive more steps to produce this frame")
            step_data = self._state_builder.get_step_data(step_idx)

        self._current_frame_idx = frame_idx
        return state, step_data, step_idx, sub_step

    def _events_on(self, frame_idx: int) -> FrameEvents:
        """ The outcome of the step this one frame completes. """
        step_idx, sub_step = divmod(frame_idx, self._expansion)
        if sub_step or step_idx <= 0 or step_idx > self._state_builder.get_step_count():
            return NO_EVENTS
        self._scan_eats_through(step_idx)
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

        expansion = self._expansion
        ate: list = []
        died: list = []
        for idx in range(((previous // expansion) + 1) * expansion, frame_idx + 1, expansion):
            events = self._events_on(idx)
            ate.extend(s_id for s_id in events.ate if s_id not in ate)
            died.extend(s_id for s_id in events.died if s_id not in died)
        if not ate and not died:
            return NO_EVENTS
        return FrameEvents(ate=tuple(ate), died=tuple(died))

    def _snake_paths(
        self,
        state: CompleteStepState,
        step_data: Optional[LoopStepData],
        sub_step: int,
    ) -> Dict[int, np.ndarray]:
        """ Every snake's pixel path for this frame, head first. """
        paths = {}
        for s_id, body in state.snake_bodies.items():
            if not body:
                continue
            decision = tail_direction = None
            if step_data is not None:
                decision = step_data.decisions.get(s_id)
                tail_direction = step_data.tail_directions.get(s_id)
            paths[s_id] = self._body_pixels(body, decision, tail_direction, sub_step)
        return paths

    def _in_bounds(self, points: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
        height, width = shape
        xs, ys = points[:, 0], points[:, 1]
        return (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)

    def _compose(
        self,
        base: np.ndarray,
        food_fill: Any,
        snake_fills: Dict[int, Tuple[Any, Any]],
        state: CompleteStepState,
        paths: Dict[int, np.ndarray],
    ) -> np.ndarray:
        """ Paint a canvas from the state. Works for both an RGB and a value canvas. """
        canvas = base.copy()
        shape = canvas.shape[:2]

        if state.food:
            food = np.array(list(state.food), dtype=np.intp) * self._expansion
            canvas[food[:, 1], food[:, 0]] = food_fill

        for s_id, points in paths.items():
            in_bounds = self._in_bounds(points, shape)
            head_in_bounds = bool(in_bounds[0])
            xs, ys = points[:, 0], points[:, 1]
            if not in_bounds.all():
                # A snake that died moving into a wall can stick out past the edge.
                xs, ys = xs[in_bounds], ys[in_bounds]
                if xs.size == 0:
                    continue

            head_fill, body_fill = snake_fills[s_id]
            canvas[ys, xs] = body_fill
            if head_in_bounds:
                canvas[ys[0], xs[0]] = head_fill

        return canvas

    # ------------------------------------------------------------------- glow

    def _scan_eats_through(self, step_idx: int):
        """ Extend the eat index so every step before `step_idx` has been seen.

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

    def _glow_intensity(self, s_id: int, path_len: int, step_idx: int, sub_step: int):
        """ Light on each pixel of a snake's path, 0 where the glow does not reach. """
        glow = self._glow
        eats = self._eat_steps.get(s_id)
        if not eats:
            return None

        glow_px = max(1, int(round(glow.length * self._expansion)))
        px_per_step = glow.speed * self._expansion
        # An eat on step e lands on the frame that completes it, so its pulse
        # starts at the head on step e + 1 and runs away from the head from there.
        oldest = step_idx - 1 - (path_len + glow_px) / px_per_step
        active = eats[bisect_left(eats, oldest):bisect_left(eats, step_idx)]
        if not active:
            return None

        fronts = (np.asarray(active, dtype=np.float64) + 1.0)
        fronts = px_per_step * (step_idx - fronts) + glow.speed * sub_step
        offsets = np.arange(glow_px, dtype=np.intp)
        # Index 0 of a path is the head, so the pulse front is the *largest*
        # index it covers and the taper runs back down towards the head.
        indices = np.rint(fronts)[:, None].astype(np.intp) - offsets[None, :]
        taper = (1.0 - offsets / glow_px) ** glow.falloff

        reaches = (indices >= 0) & (indices < path_len)
        if not reaches.any():
            return None
        flat_idx = indices[reaches]
        flat_val = np.broadcast_to(taper, indices.shape)[reaches]

        light = np.zeros(path_len, dtype=np.float32)
        if flat_idx.size:
            # Overlapping pulses keep the brightest value: writing in ascending
            # order of brightness leaves the maximum in place.
            order = np.argsort(flat_val, kind="stable")
            light[flat_idx[order]] = flat_val[order]
        return light

    def _apply_glow(self, pixels: np.ndarray, paths: Dict[int, np.ndarray], step_idx: int, sub_step: int):
        self._scan_eats_through(step_idx)
        shape = pixels.shape[:2]
        for s_id, points in paths.items():
            light = self._glow_intensity(s_id, len(points), step_idx, sub_step)
            if light is None:
                continue
            lit = light > 0.0
            lit &= self._in_bounds(points, shape)
            if not lit.any():
                continue
            xs, ys = points[lit, 0], points[lit, 1]
            amount = light[lit][:, None]
            base = pixels[ys, xs].astype(np.float32)
            glow_rgb = self._glow_rgb[s_id]
            pixels[ys, xs] = np.clip(base + amount * (glow_rgb - base), 0, 255).astype(np.uint8)

    def _body_pixels(
        self,
        body: Deque[Coord],
        decision: Optional[Coord],
        tail_direction: Optional[Coord],
        sub_step: int,
    ) -> np.ndarray:
        """ Pixel path covered by a snake, ordered from head to tail.

        The body is a chain of adjacent cells, so expanding it by `expansion` and
        filling the gaps along each segment gives a path whose index is the
        distance from the head in pixels. Within a step the head runs ahead of
        that path and the tail trails behind it, both by `sub_step` pixels.
        """
        expansion = self._expansion
        cells = np.array(body, dtype=np.intp)

        if len(cells) > 1:
            segment_dirs = cells[1:] - cells[:-1]
            offsets = np.arange(expansion, dtype=np.intp)
            points = cells[:-1, None, :] * expansion + segment_dirs[:, None, :] * offsets[None, :, None]
            points = points.reshape(-1, 2)
            points = np.concatenate((points, cells[-1:] * expansion))
        else:
            points = cells * expansion

        if sub_step and decision is not None and tuple(decision) != (0, 0):
            # Head pixels ahead of the last known cell, new head first.
            lead = np.arange(sub_step, 0, -1, dtype=np.intp)[:, None]
            points = np.concatenate((cells[0] * expansion + np.array(decision, dtype=np.intp) * lead, points))

        if sub_step and tail_direction is not None and tuple(tail_direction) != (0, 0):
            # The tail only trails if it actually moved; a snake that grew keeps it.
            points = points[:max(1, len(points) - sub_step)]

        return points
