import logging
import numpy as np

from bisect import bisect_left
from pathlib import Path
from typing import Any, Deque, Dict, Optional, Tuple

from snake_sim.environment.types import (
    Coord,
    CompleteStepState,
    EnvMetaData,
    LoopStepData,
)
from snake_sim.loop_observers.state_builder_observer import StateBuilderObserver
from snake_sim.map_utils.general import expand_map
from snake_sim.render.interfaces.frame_producer_interface import ICellGridProducer
from snake_sim.render.producers.base_frame_producer import BaseFrameProducer
from snake_sim.render.types import CellScheme, GlowConfig

log = logging.getLogger(Path(__file__).stem)


class GridFrameProducer(BaseFrameProducer, ICellGridProducer):
    """ Draws the plain one-pixel-per-cell grid rendering.

    Builds every frame from scratch out of the simulation state, so a frame is a
    pure function of (step, sub-step) and seeking anywhere is just as correct as
    playing forward. `expansion` scales the grid up and spends the extra pixels
    on interpolating snake movement: with expansion 4 a snake advances one pixel
    per frame over four frames instead of jumping a whole cell at once, which is
    also why it doubles as the producer's frames per step.

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
        super().__init__(state_builder, frames_per_step=expansion, random_colors=random_colors)
        self._expansion = self._frames_per_step
        self._glow = glow if (glow is not None and glow.enabled) else None
        self._cell_scheme: Optional[CellScheme] = None
        self._base_rgb: Optional[np.ndarray] = None
        self._base_values: Optional[np.ndarray] = None
        self._food_rgb: Optional[np.ndarray] = None
        self._food_value: int = 0
        self._snake_rgb: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self._snake_values: Dict[int, Tuple[int, int]] = {}
        self._glow_rgb: Dict[int, np.ndarray] = {}

    # ------------------------------------------------------------------ setup

    def _build(self, env: EnvMetaData) -> None:
        lut = self._lut
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
        self._cell_scheme = CellScheme(
            free=env.free_value,
            food=env.food_value,
            blocked=env.blocked_value,
            color_map=self._color_map,
        )
        if self._glow is not None:
            # The front of the pulse sits just past the head colour, towards white.
            for s_id, (head, _) in self._snake_rgb.items():
                head_f = head.astype(np.float32)
                self._glow_rgb[s_id] = head_f + self._glow.brightness * (255.0 - head_f)

        height, width = self._base_rgb.shape[:2]
        self._size = (width, height)

    @property
    def cell_scheme(self) -> CellScheme:
        self._try_init()
        return self._cell_scheme

    # ----------------------------------------------------------------- frames

    def _draw(
        self,
        state: CompleteStepState,
        step_data: Optional[LoopStepData],
        step_idx: int,
        sub_step: int,
    ) -> np.ndarray:
        paths = self._snake_paths(state, step_data, sub_step)
        pixels = self._compose(self._base_rgb, self._food_rgb, self._snake_rgb, state, paths)
        if self._glow is not None:
            self._apply_glow(pixels, paths, step_idx, sub_step)
        return pixels

    def get_cell_grid(self, frame_idx: int) -> np.ndarray:
        # No glow here: a cell value says what a cell *is*, and the glow is a
        # colour on top of that, not a different kind of cell.
        state, step_data, _, sub_step = self._resolve(frame_idx)
        paths = self._snake_paths(state, step_data, sub_step)
        return self._compose(self._base_values, self._food_value, self._snake_values, state, paths)

    # ------------------------------------------------------------------ guts

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
        self._scan_steps_through(step_idx)
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
