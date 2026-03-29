import evdev
import math
import threading
from snake_sim.snakes.input.input_provider_interface import IInputProvider

class EvdevPointerProvider(IInputProvider):
    """ An input provider that uses the evdev library to read input from a pointer device (e.g. mouse) and converts it to an angle for the snake to move towards. """

    def __init__(self, device_path: str):
        self._device = evdev.InputDevice(device_path)
        self._last_x = None
        self._last_y = None
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self):
        for event in self._device.read_loop():
            if event.type == evdev.ecodes.EV_REL:
                if event.code == evdev.ecodes.REL_X:
                    self._last_x = event.value
                elif event.code == evdev.ecodes.REL_Y:
                    self._last_y = event.value

    def get_angle(self) -> float:
        if self._last_x is None or self._last_y is None:
            return 0.0
        angle = math.atan2(-self._last_y, self._last_x) # Invert y-axis for typical screen coordinates
        return angle