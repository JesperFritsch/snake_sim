

from snake_sim.loop_observers.map_builder_observer import MapBuilderObserver
from snake_sim.render.interfaces.renderer_interface import IRenderer


def renderer_factory(renderer_type: str, map_builder: MapBuilderObserver) -> IRenderer:
    if renderer_type == "window":
        from snake_sim.render.pygame_render import PygameRenderer
        return PygameRenderer(map_builder)
    elif renderer_type == "terminal":
        from snake_sim.render.terminal_render import TerminalRenderer
        return TerminalRenderer(map_builder)
    else:
        raise ValueError(f"Unknown renderer type: {renderer_type}")