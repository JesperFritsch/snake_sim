
import time
import logging

from pathlib import Path
from dataclasses import dataclass
from typing import Set, Iterable, Tuple
from threading import Thread, Event

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
            stop_flag: Synchronized = None,
            state_builder: StateBuilderObserver = None
        ):
        self._stop_flag: Synchronized = stop_flag
        self._state_builder: StateBuilderObserver = state_builder
        self._renderer: IRenderer = renderer
        self._stop_event: Event = Event()
        self._loop_thread: Thread = Thread(target=self._loop)
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
        self._multi_steps_key = Key.shift_l
        self._toggel_paus_key = Key.space
        self._save_state_key = Key.enter
        self._quit_keys = [Key.ctrl_l, KeyCode.from_char('c')]
        self._key_combs = [
            self._forward_key,
            self._backwards_key,
            self._fast_key,
            self._multi_steps_key,
            self._toggel_paus_key,
            self._save_state_key,
            self._quit_keys
        ]
        self._listener = None

    def _loop(self):
        self._paused = False
        try:
            while ((not (self._stop_flag and self._stop_flag.value))
                    and self._running
                    and self._renderer.is_running()
                    and not self._stop_event.is_set()):

                time.sleep(0.001) # dont use all CPU

                if self._pressed(keys=self._quit_keys):
                    print("Quit keys pressed, stopping render loop")
                    self._stop_event.set()
                if self._down_event(key=self._toggel_paus_key):
                    self._paused = not self._paused
                if self._down_event(key=self._save_state_key):
                    if self._state_builder is not None:
                        step_idx = self._renderer.get_current_step_idx()
                        print(f"Saving state at step {step_idx}")
                        state = self._state_builder.get_state(step_idx)
                        log.info("Current state:\n%s", state)
                    else:
                        log.warning("No state builder attached, cannot save state")

                if self._down_event(self._forward_key):
                    self._render_frame = True
                    self._fps = self._base_fps
                elif self._down_event(self._backwards_key):
                    self._render_frame = True
                    self._frame_step_direction = False
                    self._fps = self._base_fps

                forward_pressed = self._pressed(self._forward_key)
                backward_pressed = self._pressed(self._backwards_key)

                if self._pressed(key=self._multi_steps_key):
                    self._frame_step_size = 10
                else:
                    self._frame_step_size = 1

                if forward_pressed or backward_pressed:
                    if self._pressed(key=self._fast_key):
                        self._fps = self._base_fps * 20
                        if backward_pressed:
                            self._frame_step_direction = False

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

                if self._paused:
                    self._fps = 0
                else:
                    self._fps = self._base_fps

        finally:
            self.stop()
            if self._stop_flag is not None:
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
        self._loop_thread.start()

    def stop(self) -> None:
        """Stop the listener and mark controller as stopped."""
        if not self._running:
            return
        self._running = False
        log.debug("Stopping render loop")
        self._renderer.close()
        if self._listener:
            self._listener.stop()
        self._stop_event.set()

    def join(self):
        """Wait for the render loop thread to finish.

        This join implementation uses short-time joins so a KeyboardInterrupt
        (Ctrl+C) in the main thread will be delivered promptly. On
        KeyboardInterrupt we attempt to stop the loop and re-raise the
        exception so callers (for example `main`) can handle it.
        """
        # If a timeout is provided by callers in the future, they can still
        # call thread.join(timeout) directly. Here we implement the
        # interruption-friendly no-timeout join.
        try:
            while self._loop_thread.is_alive():
                try:
                    # join in short slices so KeyboardInterrupt can be handled
                    self._loop_thread.join(timeout=0.1)
                except KeyboardInterrupt:
                    log.info("KeyboardInterrupt received in join(); stopping render loop")
                    # Best-effort stop and propagate the exception to caller
                    try:
                        self.stop()
                    except Exception:
                        pass
                    raise
        except KeyboardInterrupt:
            # Re-raise to let callers (like main) handle cleanup
            raise