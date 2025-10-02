
import numpy as np
from abc import ABC, abstractmethod


class IRenderer(ABC):
    """ Interface for renderers, it renders the simulation state to the screen. """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def render(map_buffer: np.ndarray):
        pass