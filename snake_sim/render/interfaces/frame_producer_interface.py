import numpy as np

from abc import ABC, abstractmethod
from typing import Tuple

from snake_sim.render.types import CellScheme, Frame, Palette


class IFrameProducer(ABC):
    """ Produces finished frames for a simulation run.

    Everything about *how* a frame looks lives behind this interface: palettes,
    sub-step interpolation, effects, antialiasing, the lot. Consumers are only
    responsible for getting frames somewhere useful - onto a screen, into a video
    encoder - and must not need to know what a snake or a food value is.

    What a consumer legitimately does need comes back alongside the pixels as
    `Frame.info`: where the frame sits in the run, the palette, and the events
    that landed on it. That is what lets a consumer play a sound or caption a
    frame without reaching back into the simulation itself.

    Frames are addressed by frame index, which is the producer's own timeline:
    `frames_per_step` frames map onto one simulation step, so frame index
    `step_idx * frames_per_step` is the frame where that step has just been
    applied. Producers are free to emit more than one frame per step in order to
    animate the motion in between.

    Contract for the returned pixel buffers:
      - shape (height, width, 3), dtype uint8, C-contiguous, RGB order
      - (width, height) matches `size`, which is fixed once the producer is ready
      - the buffer belongs to the caller; producers hand out a fresh array each
        call so a consumer can queue frames without them changing underneath it

    Frame lookup mirrors the loop observers and raises:
      - NoMoreSteps   - the run is still live, the frame may exist later
      - StopIteration - the run has ended, the frame will never exist
      - CurrentIsFirst - asked for a frame before the start of the run
    """

    @property
    @abstractmethod
    def size(self) -> Tuple[int, int]:
        """ (width, height) in pixels of every buffer this producer emits. """

    @property
    @abstractmethod
    def frames_per_step(self) -> int:
        """ How many frames this producer emits per simulation step. """

    @property
    @abstractmethod
    def palette(self) -> Palette:
        """ The colors this producer paints with. """

    @abstractmethod
    def is_ready(self) -> bool:
        """ True once start data has arrived and `size` is known. """

    @abstractmethod
    def wait_until_ready(self, timeout: float | None = None) -> bool:
        """ Block until `is_ready()`. Returns False if it timed out. """

    @abstractmethod
    def get_frame(self, frame_idx: int) -> Frame:
        """ Produce the frame at `frame_idx`. """

    @abstractmethod
    def get_frame_for_step(self, step_idx: int) -> Frame:
        """ Produce the frame where simulation step `step_idx` has just landed. """

    @abstractmethod
    def get_max_frame_idx(self) -> int:
        """ Highest frame index currently producible. Grows while a run is live. """

    @abstractmethod
    def get_max_step_idx(self) -> int:
        """ Highest simulation step currently available. """

    @abstractmethod
    def get_current_frame_idx(self) -> int:
        """ Frame index of the most recently produced frame. """

    @abstractmethod
    def get_current_step_idx(self) -> int:
        """ Simulation step of the most recently produced frame. """


class ICellGridProducer(IFrameProducer):
    """ A producer whose frames are a grid of discrete cells.

    Renderers that draw characters rather than pixels need to know what a cell
    *is*, not what colour it ended up. Only a producer that genuinely renders
    cell-by-cell can answer that - one drawing antialiased sub-pixel motion has
    no per-cell answer to give - so it is deliberately not part of IFrameProducer.
    """

    @property
    @abstractmethod
    def cell_scheme(self) -> CellScheme:
        """ The cell values this producer's grids are built from. """

    @abstractmethod
    def get_cell_grid(self, frame_idx: int) -> np.ndarray:
        """ The raw cell values for `frame_idx`, shaped (height, width).

        The same content as `get_frame` before any colour is applied. Raises the
        same exceptions.
        """
