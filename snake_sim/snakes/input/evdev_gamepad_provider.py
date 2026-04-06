import evdev
import evdev.ecodes as ec
import math
import threading
from snake_sim.snakes.input.input_provider_interface import IInputProvider

class EvdevGamepadProvider(IInputProvider):
    """ An input provider that uses the evdev library to read input from a pointer device (e.g. mouse) and converts it to an angle for the snake to move towards. """

    def __init__(self, device_path: str):
        self._device = evdev.InputDevice(device_path)
        self._last_x = None
        self._last_y = None
        self._dev_capabilities = self._device.capabilities()
        self._abs_infos = dict(self._dev_capabilities[ec.EV_ABS])
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self):
        def normalize(value, min_val, max_val):
            return 2 * (value - min_val) / (max_val - min_val) - 1
        x_abs_info = self._abs_infos[ec.ABS_X]
        y_abs_info = self._abs_infos[ec.ABS_Y]
        for event in self._device.read_loop():
            if event.type == ec.EV_ABS:
                self._got_event = True
                if event.code == ec.ABS_X:
                    self._last_x = normalize(event.value, x_abs_info.min, x_abs_info.max)
                elif event.code == ec.ABS_Y:
                    self._last_y = normalize(event.value, y_abs_info.min, y_abs_info.max)


    def get_angle(self) -> float:
        if self._last_x is None or self._last_y is None:
            return None
        angle = math.atan2(-self._last_y, self._last_x) # Invert y-axis for typical screen coordinates
        return angle