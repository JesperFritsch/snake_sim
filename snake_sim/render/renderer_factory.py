
from snake_sim.render.interfaces.frame_producer_interface import (
    ICellGridProducer,
    IFrameProducer,
)
from snake_sim.render.interfaces.renderer_interface import IRenderer


def renderer_factory(renderer_type: str, frame_producer: IFrameProducer) -> IRenderer:
    if renderer_type == "window":
        from snake_sim.render.pygame_render import PygameRenderer
        return PygameRenderer(frame_producer)
    elif renderer_type == "terminal":
        if not isinstance(frame_producer, ICellGridProducer):
            raise ValueError(
                f"The 'terminal' renderer draws cells as characters and needs an "
                f"ICellGridProducer, got {type(frame_producer).__name__}"
            )
        from snake_sim.render.terminal_render import TerminalRenderer
        return TerminalRenderer(frame_producer)
    else:
        raise ValueError(f"Unknown renderer type: {renderer_type}")
