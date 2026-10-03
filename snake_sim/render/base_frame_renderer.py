from abc import abstractmethod

from snake_sim.environment.types import CurrentIsFirst, NoMoreSteps
from snake_sim.render.interfaces.frame_producer_interface import IFrameProducer
from snake_sim.render.interfaces.renderer_interface import IRenderer


class BaseFrameRenderer(IRenderer):
    """ Index plumbing shared by renderers driven by an IFrameProducer.

    Subclasses only implement `_render_at`, pulling whichever of the producer's
    outputs suits them and putting it on their output.
    """

    def __init__(self, frame_producer: IFrameProducer):
        super().__init__()
        self._producer = frame_producer

    @abstractmethod
    def _render_at(self, frame_idx: int):
        """ Fetch and show the frame at `frame_idx`. """

    def _try_render_at(self, frame_idx: int):
        try:
            self._render_at(frame_idx)
        except (StopIteration, NoMoreSteps, CurrentIsFirst):
            return

    def render_frame(self, frame_idx: int):
        self._try_render_at(frame_idx)

    def render_step(self, step_idx: int):
        self._try_render_at(step_idx * self._producer.frames_per_step)

    def render_first_frame(self):
        self._try_render_at(0)

    def render_middle_frame(self):
        self._try_render_at(self._producer.get_max_frame_idx() // 2)

    def render_last_frame(self):
        self._try_render_at(self._producer.get_max_frame_idx())

    def get_current_map_idx(self) -> int:
        return self._producer.get_current_frame_idx()

    def get_current_step_idx(self) -> int:
        return self._producer.get_current_step_idx()
