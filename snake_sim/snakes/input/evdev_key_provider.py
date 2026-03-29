import evdev
import math
import threading
from snake_sim.snakes.input.input_provider_interface import IInputProvider


class EvdevKeyProvider(IInputProvider):
    def __init__(self, device_path: str, key_mapping: dict[int, tuple[int, int]]):
        """
        key_mapping: evdev key code -> (dx, dy) direction vector in math convention.
        Example: {
            ecodes.KEY_W: (0, 1),
            ecodes.KEY_S: (0, -1),
            ecodes.KEY_A: (-1, 0),
            ecodes.KEY_D: (1, 0),
        }
        """
        self._device = evdev.InputDevice(device_path)
        self._key_mapping = key_mapping
        self._held: set[int] = set()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self):
        for event in self._device.read_loop():
            if event.type == evdev.ecodes.EV_KEY and event.code in self._key_mapping:
                with self._lock:
                    if event.value == 1:
                        self._held.add(event.code)
                    elif event.value == 0:
                        self._held.discard(event.code)

    def get_angle(self) -> float | None:
        with self._lock:
            if not self._held:
                return None
            dx = sum(self._key_mapping[code][0] for code in self._held)
            dy = -sum(self._key_mapping[code][1] for code in self._held)

        if dx == 0 and dy == 0:
            return None

        return math.atan2(dy, dx)