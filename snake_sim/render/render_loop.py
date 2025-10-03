
import time
import logging

from pathlib import Path
from dataclasses import dataclass
from typing import Set, Iterable, Tuple

from multiprocessing.sharedctypes import Synchronized
from snake_sim.render.interfaces.renderer_interface import IRenderer
from snake_sim.loop_observers.state_builder_observer import StateBuilderObserver

# Use the same headless workaround as other controllers that use pynput
from snake_sim.utils import is_headless

log = logging.getLogger(Path(__file__).stem)

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
            state_builder: StateBuilderObserver,
            stop_flag: Synchronized
        ):
        self._stop_flag: Synchronized = stop_flag
        self._state_builder: StateBuilderObserver = state_builder
        self._renderer: IRenderer = renderer
        self._frame_step_direction = True # True for forward
        self._frame_step_size = 1
        self._base_fps = config.fps
        self._fps = self._base_fps
        self._keys_pressed: Set[keyboard.Key | keyboard.KeyCode] = set()
        # Tracks combos that have already reported a down-event until released
        self._reported_downs: Set[Tuple[keyboard.Key | keyboard.KeyCode]] = set()
        self._paused = False # stepping frames
        self._running = False # loop running
        self._render_frame = False
        self._last_render_time = time.time()
        self._forward_key = Key.right
        self._backwards_key = Key.left
        self._fast_key = Key.ctrl_l
        self._super_fast_keys = [Key.shift_l, self._fast_key]
        self._toggel_paus_key = Key.space
        self._save_state_key = Key.enter
        self._quit_keys = [Key.ctrl_l, KeyCode.from_char('c')]
        self._key_combs = [
            self._forward_key,
            self._backwards_key,
            self._fast_key,
            self._super_fast_keys,
            self._toggel_paus_key,
            self._save_state_key,
            self._quit_keys
        ]
        self._listener = None

    def _loop(self):
        self._paused = False
        try:
            while not self._stop_flag.value and self._running:
                if self._pressed(keys=self._quit_keys):
                    self._running = False
                if self._down_event(key=self._toggel_paus_key):
                    self._paused = not self._paused
                if self._down_event(key=self._save_state_key):
                    print("Statesave not implemented")

                if self._down_event(self._forward_key):
                    self._render_frame = True
                    self._fps = self._base_fps
                elif self._down_event(self._backwards_key):
                    self._render_frame = True
                    self._frame_step_direction = False
                    self._fps = self._base_fps

                forward_pressed = self._pressed(self._forward_key)
                backward_pressed = self._pressed(self._backwards_key)

                if forward_pressed or backward_pressed:
                    if self._pressed(keys=self._super_fast_keys):
                        if backward_pressed:
                            self._frame_step_direction = False
                        self._fps = self._base_fps * 20
                        self._frame_step_size = 5
                    elif self._pressed(key=self._fast_key):
                        if backward_pressed:
                            self._frame_step_direction = False
                        self._fps = self._base_fps * 20
                        self._frame_step_size = 1

                if self._render_frame:
                    current_frame_idx = self._renderer.get_current_frame_idx()
                    next_frame_idx = current_frame_idx + (self._frame_step_size * (1 if self._frame_step_direction else -1))
                    next_frame_idx = max(next_frame_idx, 0)
                    self._renderer.render_frame(next_frame_idx)
                    self._last_render_time = time.time()
                    self._render_frame = False
                    self._frame_step_direction = True
                    self._frame_step_size = 1

                if self._fps > 0:
                    fps_duration = 1 / self._fps
                    if time.time() - self._last_render_time > fps_duration:
                        self._render_frame = True
                else:
                    time.sleep(0.01) # dont use all CPU

                if self._paused:
                    self._fps = 0
                else:
                    self._fps = self._base_fps

        finally:
            self.stop()
            self._stop_flag.value = True

    def _handle_pressed(self, key):
        self._keys_pressed.add(key)

    def _handle_released(self, key):
        # Remove from currently pressed
        self._keys_pressed.discard(key)
        # Any reported combos that include this key must be cleared so they
        # can report again on the next press
        to_remove = []
        for combo in self._reported_downs:
            if key in combo:
                to_remove.append(combo)
        for combo in to_remove:
            self._reported_downs.discard(combo)

    def _pressed(self, key: Key | KeyCode = None, keys: Iterable[Key | KeyCode] = None):
        # will return true as long as the key combination is pressed
        if keys is not None:
            result = all(k in self._keys_pressed for k in keys)
        else:
            result = key in self._keys_pressed
        return result

    def _down_event(self, key: Key | KeyCode = None, keys: Iterable[Key | KeyCode] = None):
        # Normalize parameters into a tuple representing the combo
        if keys is not None:
            try:
                combo = tuple(keys)
            except TypeError:
                combo = (keys,)
        elif key is not None:
            combo = (key,)
        else:
            return False

        # If combo currently pressed and not yet reported, report and mark it
        if all(k in self._keys_pressed for k in combo):
            if combo not in self._reported_downs:
                self._reported_downs.add(combo)
                return True
            return False

        # Combo not pressed -> ensure it's not marked as reported
        if combo in self._reported_downs:
            self._reported_downs.discard(combo)
        return False

    def start(self) -> None:
        """Start the pynput listener in a background thread."""
        if self._running:
            return
        self._running = True
        self._listener = keyboard.Listener(on_press=self._handle_pressed, on_release=self._handle_released)
        # run the listener in a dedicated thread
        self._listener.start()
        self._loop()

    def stop(self) -> None:
        """Stop the listener and mark controller as stopped."""
        if not self._running:
            return
        self._running = False
        log.debug("Stopping render loop")
        self._renderer.close()
        if self._listener:
            self._listener.stop()

    def join(self):
        self._listener.join()