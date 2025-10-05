

import sys
import time
import numpy as np
import logging

from pathlib import Path
from threading import Thread

from snake_sim.render.interfaces.renderer_interface import IRenderer
from snake_sim.loop_observers.frame_builder_observer import FrameBuilderObserver, NoMoreSteps, CurrentIsFirst
from snake_sim.map_utils.general import print_map
from snake_sim.utils import create_color_map

try:
    from colorama import init as colorama_init
    colorama_init()
except Exception:
    pass

log = logging.getLogger(Path(__file__).stem)

CSI = "\x1b["  # Control Sequence Introducer

class TerminalRenderer(IRenderer):
    """ A simple terminal renderer that prints the state to the console. """
    def __init__(self, frame_builder: FrameBuilderObserver):
        super().__init__()
        self._frame_builder = frame_builder
        self._current_step = 0
        self._current_frame = 0
        self._wait_thread = Thread(target=self._finish_init, daemon=True)
        self._wait_thread.start()
        self._written_lines = 0
        self._env_init_data = None
        self._color_map = None
        self._free_value = None
        self._food_value = None
        self._blocked_value = None

    def is_init_finished(self):
        return self._init_finished

    def _finish_init(self):
            while self._frame_builder._start_data is None:
                time.sleep(0.005)
            self._env_init_data = self._frame_builder._start_data.env_init_data
            self._color_map = create_color_map(self._env_init_data.snake_values)
            self._free_value = self._env_init_data.free_value
            self._food_value = self._env_init_data.food_value
            self._blocked_value = self._env_init_data.blocked_value
            self._init_finished = True

    def _render_frame(self, frame: np.ndarray):
        if not self.is_init_finished():
            log.debug("Skipping render frame; init not finished")
            return

        if self._written_lines > 0:
            self._move_cursor_up(self._written_lines)
        self._written_lines = print_map(
            s_map=frame,
            free_value=self._free_value,
            food_value=self._food_value,
            blocked_value=self._blocked_value,
            color_map=self._color_map
        )

    def render_step(self, step_idx: int):
        try:
            frame = self._frame_builder.get_frame_for_step(step_idx)
            self._render_frame(frame)
            self._current_step = step_idx
        except (StopIteration, NoMoreSteps, CurrentIsFirst):
            pass
        return self._current_step

    def render_frame(self, frame_idx: int):
        try:
            frame = self._frame_builder.get_frame(frame_idx)
            self._render_frame(frame)
            self._current_frame = frame_idx
        except (StopIteration, NoMoreSteps, CurrentIsFirst):
            pass
        return self._current_frame

    def get_current_frame_idx(self):
        self._current_frame = self._frame_builder.get_current_frame_idx()
        return self._current_frame

    def get_current_step_idx(self):
        self._current_step = self._frame_builder.get_current_step_idx()
        return self._current_step

    def is_running(self):
        # There is nothing to "run" but the render loop will exit if this is false
        return True

    def close(self):
        pass

    def _move_cursor_up(self, lines: int):
        sys.stdout.write(f"{CSI}{lines}A")

    def _clear_lines(self, lines: int):
        sys.stdout.write(f"{CSI}{lines}K")

    def _flush(self):
        sys.stdout.flush()
