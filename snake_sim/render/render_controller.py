
import numpy as np
import queue
from typing import List, Tuple, Any, Dict, Callable

from snake_sim.render.interfaces.renderer_interface import IRenderer

# Use the same headless workaround as other controllers that use pynput
from snake_sim.utils import is_headless

if is_headless():
    # pynput may require a display; start a virtual one in headless environments
    try:
        from pyvirtualdisplay import Display
        _display = Display(visible=False, size=(1024, 768))
        _display.start()
    except Exception:
        # If this fails, listeners may still work or tests will handle it
        _display = None

from pynput import keyboard


class Action:
    """ Represents a keyboard action, which can be a single key or a combination of keys. """
    def __init__(self, keys: Tuple[keyboard.Key], handle: Callable, restore: Callable = None):
        self._handle: Callable = handle
        self._restore: Callable = restore
        self.keys = tuple(keys)

    def __hash__(self):
        return hash(self.keys)

    def __eq__(self, other):
        if not isinstance(other, Action):
            return False
        return self.keys == other.keys
    
    def check(self, pressed_keys: set) -> bool:
        return all(k in pressed_keys for k in self.keys)
    
    def handle(self):
        self._handle()
    
    def restore(self):
        if self._restore:
            self._restore()


class KeyboardController:

    def __init__(self, renderer: IRenderer) -> None:
        self._renderer = renderer
        self._events = queue.Queue()
        self._pressed = set()
        self._listener = None
        self._running = False
        self._actions: List[Action] = [
            Action(
                (keyboard.Key.space,), 
                handle=renderer.toggle_pause
            ),
            Action(
                (keyboard.Key.left,),
                handle=lambda: renderer.step_back() if renderer.is_paused() else renderer.set_direction_backward(),
                restore=lambda: None if renderer.is_paused() else renderer.set_direction_forward()
            ),
            Action(
                (keyboard.Key.right,),
                handle=lambda: renderer.step_forward() if renderer.is_paused() else renderer.set_direction_forward(),
                restore=lambda: None if renderer.is_paused() else renderer.set_direction_backward()
            ),
        ]

    def _handle_event(self, key: Any):
        if isinstance(key, keyboard.Key) or isinstance(key, keyboard.KeyCode):
            if key not in self._pressed:
                self._pressed.add(key)
            else:
                self._pressed.remove(key)
        for action in self._actions:
            if action.check(self._pressed):
                action.handle()
            else:
                action.restore()

    def start(self) -> None:
        """Start the pynput listener in a background thread."""
        if self._running:
            return
        self._running = True
        self._listener = keyboard.Listener(on_press=self._handle_event, on_release=self._handle_event)
        # run the listener in a dedicated thread
        self._listener.start()

    def stop(self) -> None:
        """Stop the listener and mark controller as stopped."""
        if not self._running:
            return
        self._running = False
        try:
            if self._listener:
                self._listener.stop()
        except Exception:
            pass
