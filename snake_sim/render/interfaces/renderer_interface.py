
import numpy as np
from abc import ABC, abstractmethod


class IRenderer(ABC):
    """ Interface for renderers, it renders the simulation state to the screen. """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def render_step(self, step_idx: int):
        pass

    @abstractmethod
    def render_frame(self, frame_idx: int):
        pass

    @abstractmethod
    def get_current_step_idx(self) -> int:
        pass

    @abstractmethod
    def get_current_frame_idx(self) -> int:
        pass