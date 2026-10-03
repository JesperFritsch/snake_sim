import sys
import numpy as np
import logging

from pathlib import Path

from snake_sim.render.base_frame_renderer import BaseFrameRenderer
from snake_sim.render.interfaces.frame_producer_interface import ICellGridProducer
from snake_sim.map_utils.general import print_map

try:
    from colorama import init as colorama_init
    colorama_init()
except Exception:
    pass

log = logging.getLogger(Path(__file__).stem)

CSI = "\x1b["  # Control Sequence Introducer

class TerminalRenderer(BaseFrameRenderer):
    """ Prints the cell grid to the console as characters.

    Draws characters rather than pixels, so it reads the producer's cell values
    instead of its pixel buffers - a colour alone cannot say whether a cell is
    empty or a wall, and it stops being able to once effects tint it.
    """
    def __init__(self, frame_producer: ICellGridProducer):
        super().__init__(frame_producer)
        self._written_lines = 0

    def is_init_finished(self):
        return self._producer.is_ready()

    def _render_at(self, frame_idx: int):
        grid = self._producer.get_cell_grid(frame_idx)
        scheme = self._producer.cell_scheme
        if self._written_lines > 0:
            self._move_cursor_up(self._written_lines)
        self._written_lines = print_map(
            s_map=grid,
            free_value=scheme.free,
            food_value=scheme.food,
            blocked_value=scheme.blocked,
            color_map=scheme.color_map
        )

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
