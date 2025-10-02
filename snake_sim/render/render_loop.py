
from dataclasses import dataclass
from typing import Set, Iterable
from threading import Thread

from snake_sim.environment.interfaces.loop_observable_interface import ILoopObservable
from snake_sim.render.interfaces.renderer_interface import IRenderer
from snake_sim.loop_observers.frame_builder_observer import FrameBuilderObserver

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
from pynput.keyboard import Key, KeyCode

@dataclass
class RenderConfig:
    fps: int
    sound: bool


class RenderLoop:
    def __init__(self, 
            renderer: IRenderer, 
            config: RenderConfig,
            loop_observable: ILoopObservable
        ):
        self._frame_builder: FrameBuilderObserver = FrameBuilderObserver()
        self._renderer: IRenderer = renderer
        self._base_fps = config.fps
        self._fps = self._base_fps
        self._keys_pressed: Set[keyboard.Key | keyboard.KeyCode] = set() 
        self._running = False

    def _loop(self):
        while self._running:


    

    def _handle_pressed(self, key):
        self._keys_pressed.add(key)

    def _handle_released(self, key):
        self._keys_pressed.discard(key)

    def _pressed(self, key: Key | KeyCode = None, keys: Iterable[Key | KeyCode] = None):
        if keys is not None:
            return all(k in self._keys_pressed for k in keys)
        else:
            return key in self._keys_pressed

    def start(self) -> None:
        """Start the pynput listener in a background thread."""
        if self._running:
            return
        self._running = True
        self._listener = keyboard.Listener(on_press=self._handle_pressed, on_release=self._handle_released)
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