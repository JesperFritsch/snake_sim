

from snake_sim.loop_observers.frame_builder_observer import FrameBuilderObserver
from snake_sim.render.interfaces.renderer_interface import IRenderer


def renderer_factory(renderer_type: str, frame_builder: FrameBuilderObserver) -> IRenderer:
    if renderer_type == "window":
        from snake_sim.render.pygame_render import PygameRenderer
        return PygameRenderer(frame_builder)
    elif renderer_type == "terminal":
        from snake_sim.render.terminal_render import TerminalRenderer
        return TerminalRenderer(frame_builder)
    else:
        raise ValueError(f"Unknown renderer type: {renderer_type}")