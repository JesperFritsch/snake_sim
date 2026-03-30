
import math

from snake_sim.environment.interfaces.snake_strategy_interface import ISnakeStrategy
from snake_sim.environment.types import StrategyConfig, Coord
from snake_sim.snakes.input.input_provider_interface import IInputProvider

from snake_sim.snakes.input.input_utils import create_input_provider, AvailableInput
from snake_sim.snakes.input.evdev_pointer_provider import EvdevPointerProvider
from snake_sim.snakes.input.evdev_key_provider import EvdevKeyProvider
import evdev.ecodes as ec


class ManualStrategy(ISnakeStrategy):
    def __init__(self, strategy_config: StrategyConfig):
        super().__init__(strategy_config)
        # # self._input_provider = EvdevPointerProvider(device_path="/dev/input/event16")
        
        # self._input_provider = EvdevKeyProvider(device_path="/dev/input/event3", key_mapping={
        #     ec.KEY_UP: (0, -1),
        #     ec.KEY_RIGHT: (1, 0),
        #     ec.KEY_DOWN: (0, 1),
        #     ec.KEY_LEFT: (-1, 0)
        # })
        print(strategy_config)
        self._input_provider = create_input_provider(
            strategy_config.params["input_config"]
        )

        self._acc_x = 0.0
        self._acc_y = 0.0
        self._last_angle: float | None = None
        self._last_direction = None

    def get_wanted_tile(self) -> Coord:
        angle = self._input_provider.get_angle()
        if angle is None:
            if self._last_direction:
                return self._snake.get_head_coord() + self._last_direction  
            return

        if self._last_angle is not None:
            delta = abs(angle - self._last_angle)
            delta = min(delta, 2 * math.pi - delta)  # shortest arc
            if delta > math.pi / 8:  # ~22.5° threshold
                self._acc_x = 0.0
                self._acc_y = 0.0
        self._last_angle = angle

        self._acc_x += math.cos(angle)
        self._acc_y += -math.sin(angle)

        if abs(self._acc_x) >= abs(self._acc_y):
            step = Coord(1 if self._acc_x > 0 else -1, 0)
            self._acc_x -= step.x
        else:
            step = Coord(0, 1 if self._acc_y > 0 else -1)
            self._acc_y -= step.y
        self._last_direction = step
        return self._snake.get_head_coord() + step
