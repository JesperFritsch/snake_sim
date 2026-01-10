import logging
from pathlib import Path

log = logging.getLogger(Path(__file__).stem)

try:
    import pygame
except Exception:
    pygame = None

class BaseInputProvider:
    """Abstracts keyboard input for the render loop.

    Methods:
        pump(): Refresh internal key state.
        is_pressed(key: str) -> bool: Returns True while key is held.
        down_event(key: str) -> bool: Returns True exactly once when key transitions to down.
        stop(): Cleanup resources (optional in implementations).
    """
    KEY_NAMES = ["RIGHT", "LEFT", "CTRL", "SHIFT", "SPACE", "ENTER", "QUIT", "ESC"]

    def pump(self):
        pass

    def is_pressed(self, key: str=None, keys: list[str]=None) -> bool:
        return False

    def down_event(self, key: str) -> bool:
        return False

    def stop(self):
        pass

class DummyInputProvider(BaseInputProvider):
    """No-op provider used in headless or unsupported environments."""
    pass

class TerminalInputProvider(BaseInputProvider):
    """Cross-platform terminal input provider with modifier detection.

    POSIX: non-blocking raw stdin (termios + fcntl + select).
    Windows: msvcrt polling.

    Arrow escape sequences may include modifier encodings, e.g.:
        ESC [ C          -> RIGHT
        ESC [ D          -> LEFT
        ESC [ 1 ; 2 C    -> SHIFT + RIGHT
        ESC [ 1 ; 5 C    -> CTRL + RIGHT
        ESC [ 1 ; 6 C    -> SHIFT + CTRL + RIGHT

    Modifier code mapping (xterm style):
        2 Shift, 3 Alt, 4 Shift+Alt, 5 Ctrl, 6 Shift+Ctrl, 7 Alt+Ctrl, 8 Shift+Alt+Ctrl

    Standalone SHIFT/CTRL key presses do not produce characters; we infer
    modifier state only from arrow sequences containing them and keep them
    "pressed" for hold_duration. This is sufficient for multi-step (SHIFT)
    and fast (CTRL) logic in the render loop when used with navigation keys.
    """
    def __init__(self, hold_duration: float = 0.25):
        import sys, time
        self._sys = sys
        self._time = time
        self._pressed_times: dict[str, float] = {}
        self._modifier_times: dict[str, float] = {}
        self._reported_down: set[str] = set()
        self._hold_duration = hold_duration
        self._last_pump = time.time()
        # platform specifics
        self._use_msvcrt = False
        try:
            import msvcrt  # type: ignore
            self._msvcrt = msvcrt
            if sys.platform.startswith("win"):
                self._use_msvcrt = True
        except Exception:
            self._msvcrt = None
        # Setup POSIX non-blocking mode
        if not self._use_msvcrt:
            try:
                import termios, tty, fcntl, os, select
                self._termios = termios
                self._tty = tty
                self._fcntl = fcntl
                self._os = os
                self._select = select
                self._fd = sys.stdin.fileno()
                self._old_settings = termios.tcgetattr(self._fd)
                tty.setcbreak(self._fd)
                # Set non-blocking
                flags = fcntl.fcntl(self._fd, fcntl.F_GETFL)
                fcntl.fcntl(self._fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
            except Exception as e:
                log.warning("Failed to configure non-blocking terminal input: %s", e)
                self._fd = None

    def pump(self):
        now = self._time.time()
        # Collect new keys
        new_keys = []
        if self._use_msvcrt and self._msvcrt:
            try:
                while self._msvcrt.kbhit():
                    ch = self._msvcrt.getch()
                    if ch in (b"\x00", b"\xe0"):
                        # Special key prefix; get next
                        ch2 = self._msvcrt.getch()
                        code_map = {b"M": "RIGHT", b"K": "LEFT"}
                        mapped = code_map.get(ch2)
                        if mapped:
                            new_keys.append(mapped)
                    else:
                        k = ch.decode(errors="ignore")
                        mapped = self._map_char(k)
                        if mapped:
                            new_keys.append(mapped)
            except Exception:
                pass
        else:
            if self._fd is not None:
                try:
                    # Use select for readability without consuming full buffer repeatedly
                    if self._select.select([self._fd], [], [], 0)[0]:
                        data = self._sys.stdin.read()
                        if data:
                            new_keys.extend(self._parse_stream(data))
                except Exception:
                    pass
        for k in new_keys:
            if k in ("RIGHT", "LEFT", "SPACE", "ENTER", "QUIT", "ESC"):
                if k not in self._pressed_times:
                    self._pressed_times[k] = now
            elif k in ("SHIFT", "CTRL"):
                # Modifiers tracked separately
                self._modifier_times[k] = now
        # Expire non-modifier holds (arrow keys single-shot by design)
        to_remove = [k for k, t in self._pressed_times.items() if now - t > self._hold_duration]
        for k in to_remove:
            self._pressed_times.pop(k, None)
        # Expire modifier synthetic holds
        mod_remove = [k for k, t in self._modifier_times.items() if now - t > self._hold_duration]
        for k in mod_remove:
            self._modifier_times.pop(k, None)
        self._last_pump = now

    def _parse_stream(self, data: str) -> list[str]:
        keys: list[str] = []
        i = 0
        length = len(data)
        while i < length:
            ch = data[i]
            if ch == "\x1b":
                j = i + 1
                if j < length and data[j] == '[':
                    k = j + 1
                    while k < length and not data[k].isalpha():
                        k += 1
                    if k < length and data[k] in 'ABCD':
                        final = data[k]
                        params = data[j+1:k]  # content between '[' and final letter
                        modifier_code = None
                        if ';' in params:
                            # Last number usually modifier
                            parts = params.split(';')
                            try:
                                modifier_code = int(parts[-1])
                            except ValueError:
                                modifier_code = None
                        arrow_map = {'C': 'RIGHT', 'D': 'LEFT'}
                        mapped = arrow_map.get(final)
                        if mapped:
                            keys.append(mapped)
                            if modifier_code is not None:
                                if modifier_code in {2,4,6,8}:  # Shift included
                                    keys.append('SHIFT')
                                if modifier_code in {5,6,7,8}:  # Ctrl included
                                    keys.append('CTRL')
                        i = k + 1
                        continue
                # Single ESC (not CSI arrow) -> ESC key
                keys.append('ESC')
                i += 1
            else:
                mapped = self._map_char(ch)
                if mapped:
                    keys.append(mapped)
                i += 1
        return keys

    def _map_char(self, ch: str):
        char_map = {
            " ": "SPACE",
            "\n": "ENTER",
            "\r": "ENTER",
            "q": "QUIT",
            "Q": "QUIT",
            "\x1b": "ESC",
        }
        return char_map.get(ch)

    def is_pressed(self, key: str=None, keys: list[str]=None) -> bool:
        if keys is not None:
            if all(k in self._pressed_times or k in self._modifier_times for k in keys):
                return True
            return False
        else:
            if key in self._pressed_times:
                return True
            if key in self._modifier_times:
                return True
            return False

    def down_event(self, key: str) -> bool:
        # Modifiers do not produce explicit down events (inferred)
        if key in ("SHIFT", "CTRL"):
            return False
        if key in self._pressed_times and key not in self._reported_down:
            self._reported_down.add(key)
            # Remove so another press can be detected after repeat delay
            self._pressed_times.pop(key, None)
            return True
        return False

    def end_frame(self):
        # Clear edge reports only; pressed_times already pruned for down_event removals
        self._reported_down.clear()

    def stop(self):
        if not self._use_msvcrt and getattr(self, "_fd", None) is not None:
            try:
                self._termios.tcsetattr(self._fd, self._termios.TCSADRAIN, self._old_settings)
            except Exception:
                pass

class PygameInputProvider(BaseInputProvider):
    """Input provider backed by Pygame's event & key state system.

    This relies on a Pygame display already being initialized elsewhere.
    It does not consume (drain) window events beyond pumping, so other
    components may still read them if needed.
    """
    def __init__(self):
        if pygame is None:
            raise RuntimeError("pygame not available for PygameInputProvider")
        # Ensure initialization (safe if already done)
        if not pygame.get_init():
            pygame.init()
        self._prev = pygame.key.get_pressed()
        self._curr = self._prev
        self._reported_down = set()
        self._key_map = {
            "RIGHT": pygame.K_RIGHT,
            "LEFT": pygame.K_LEFT,
            "CTRL": pygame.K_LCTRL,
            "SHIFT": pygame.K_LSHIFT,
            "SPACE": pygame.K_SPACE,
            "ENTER": pygame.K_RETURN,
            "C": pygame.K_c,
            "E": pygame.K_e,
            "S": pygame.K_s,
            "M": pygame.K_m,
        }

    def pump(self):
        try:
            pygame.event.pump()
        except Exception:
            # If window closed, key presses won't matter; keep previous state.
            pass
        try:
            self._curr = pygame.key.get_pressed()
        except Exception:
            # If window closed, key presses won't matter; keep previous state.
            pass

    def is_pressed(self, key: str=None, keys: list[str]=None) -> bool:

        if keys is not None:
            for k in keys:
                code = self._key_map.get(k)
                if code is None or not self._curr[code]:
                    return False
            return True
        code = self._key_map.get(key)
        if code is None:
            return False
        return bool(self._curr[code])

    def down_event(self, key: str) -> bool:
        code = self._key_map.get(key)
        if code is None:
            return False
        pressed_now = bool(self._curr[code])
        pressed_prev = bool(self._prev[code])
        if pressed_now and not pressed_prev:
            # Edge trigger; record so multiple calls this cycle still return True only once
            if code not in self._reported_down:
                self._reported_down.add(code)
                return True
        return False

    def end_frame(self):
        # Clear reported downs for next frame and advance previous state
        self._reported_down.clear()
        self._prev = self._curr

    def stop(self):
        # Nothing special; we do not own the display
        pass

__all__ = [
    "BaseInputProvider",
    "DummyInputProvider",
    "PygameInputProvider",
    "TerminalInputProvider",
]
