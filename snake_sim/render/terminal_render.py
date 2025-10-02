

from snake_sim.render.interfaces.renderer_interface import IRenderer


class TerminalRenderer(IRenderer):
    """ A simple terminal renderer that prints the state to the console. """
    def __init__(self, fps: int):
        super().__init__(fps)
        