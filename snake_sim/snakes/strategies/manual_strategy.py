
import math

from snake_sim.environment.interfaces.snake_strategy_interface import ISnakeStrategy
from snake_sim.environment.types import StrategyConfig, Coord
from snake_sim.snakes.input.input_provider_interface import IInputProvider


from snake_sim.snakes.input.evdev_input_provider import EvdevPointerProvider


class ManualStrategy(ISnakeStrategy):
    """ This strategy does not make any decisions, it converts the user provided angle
    to a next tile and returns it.
    """

    def __init__(self, strategy_config: StrategyConfig):
        super().__init__(strategy_config)
        self._input_provider: IInputProvider = EvdevPointerProvider(device_path="/dev/input/event16")


    def get_wanted_tile(self) -> Coord:
        angle = self._input_provider.get_angle()
        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)
        dir_x = 0
        dir_y = 0
        if abs(cos_angle) > abs(sin_angle):
            dir_x = round(cos_angle)
        else:
            dir_y = -round(sin_angle)
        print(dir_x, dir_y)
        return self._snake.get_head_coord() + Coord(dir_x, dir_y)
