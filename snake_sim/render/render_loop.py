
import time
import logging

from pathlib import Path
from dataclasses import dataclass
from typing import Iterable
from threading import Thread, Event
from multiprocessing.sharedctypes import Synchronized

from snake_sim.storing.state_storer import save_step_state
from snake_sim.render.interfaces.renderer_interface import IRenderer
from snake_sim.loop_observers.state_builder_observer import StateBuilderObserver
from snake_sim.render.input_provider import (
    BaseInputProvider,
    DummyInputProvider,
    PygameInputProvider,
    TerminalInputProvider,
)
from snake_sim.render.terminal_render import TerminalRenderer
from snake_sim.utils import is_headless

log = logging.getLogger(Path(__file__).stem)

@dataclass
class RenderConfig:
    fps: int
    sound: bool


class RenderLoop:
    """Main render control loop with cross-platform window-focused input.

    Key bindings (window must be focused):
        RIGHT Arrow  : step/render forward continuously while held
        LEFT Arrow   : step/render backward continuously while held
        CTRL         : when held with an arrow, fast mode (higher FPS multiplier)
        SHIFT        : increases frame step size to 10 while held
        SPACE        : toggle pause (freeze auto rendering; manual stepping allowed)
        ENTER        : save current state (if state builder attached)
        CTRL + C     : quit render loop (mirrors previous global shortcut)

    On Wayland and Windows this uses Pygame's event system instead of global
    hooks, avoiding failure modes of pynput/keyboard libraries under Wayland.
    """

    def __init__(
        self,
        renderer: IRenderer,
        config: RenderConfig,
        stop_flag: Synchronized = None,
        state_builder: StateBuilderObserver = None,
        input_provider: BaseInputProvider | None = None,
    ):
        self._stop_flag: Synchronized = stop_flag
        self._state_builder: StateBuilderObserver = state_builder
        self._renderer: IRenderer = renderer
        self._stop_event: Event = Event()
        self._loop_thread: Thread = Thread(target=self._loop)
        self._frame_step_direction = True  # True for forward
        self._frame_step_size = 1
        self._base_fps = config.fps
        self._fps = self._base_fps
        self._paused = False  # stepping frames
        self._running = False  # loop running
        self._render_frame = False
        self._last_render_time = time.time()

        # Key identifiers (mapped inside provider)
        self._forward_key = "RIGHT"
        self._backwards_key = "LEFT"
        self._fast_key = "CTRL"
        self._multi_steps_key = "SHIFT"
        self._toggel_paus_key = "SPACE"
        self._save_state_key = "ENTER"
        self._quit_combo = ("CTRL", "C")

        # Choose input provider
        if input_provider is not None:
            self._input = input_provider
            log.debug("RenderLoop input provider explicitly supplied: %s", type(self._input).__name__)
        else:
            # Selection priority based on renderer instance:
            # 1. TerminalRenderer -> TerminalInputProvider (TTY interaction)
            # 2. PygameRenderer   -> PygameInputProvider
            # 3. Fallback order: TerminalInputProvider (if TTY) else Dummy
            import sys
            chosen = None
            if isinstance(self._renderer, TerminalRenderer):
                try:
                    self._input = TerminalInputProvider()
                    chosen = "terminal (forced by renderer type)"
                except Exception as e:
                    log.warning("TerminalInputProvider failed (%s); falling back", e)
                    self._input = DummyInputProvider()
                    chosen = "dummy (terminal fallback)"
            else:
                # Try pygame first if not headless
                if not is_headless():
                    try:
                        self._input = PygameInputProvider()
                        chosen = "pygame"
                    except Exception:
                        chosen = None
                if chosen is None:
                    if sys.stdin.isatty():
                        try:
                            self._input = TerminalInputProvider()
                            chosen = "terminal (tty)"
                        except Exception:
                            chosen = None
                if chosen is None:
                    self._input = DummyInputProvider()
                    chosen = "dummy"
            log.debug("RenderLoop input provider selected: %s", chosen)

    def _loop(self):
        self._paused = False
        try:
            while (
                (not (self._stop_flag and self._stop_flag.value))
                and self._running
                and self._renderer.is_running()
                and not self._stop_event.is_set()
            ):
                time.sleep(0.005)  # light sleep to reduce CPU
                # Update input state
                self._input.pump()

                # Quit detection
                if isinstance(self._input, TerminalInputProvider):
                    # Terminal quit: 'q' or ESC
                    if self._input.down_event("QUIT") or self._input.down_event("ESC"):
                        log.info("Terminal quit key pressed; stopping render loop")
                        self._stop_event.set()
                else:
                    if all(self._input.is_pressed(k) for k in self._quit_combo):
                        log.info("Quit combo pressed; stopping render loop")
                        self._stop_event.set()

                # Toggle pause
                if self._input.down_event(self._toggel_paus_key):
                    self._paused = not self._paused

                # Save state
                if self._input.down_event(self._save_state_key):
                    if self._state_builder is not None:
                        step_idx = self._renderer.get_current_step_idx()
                        state = self._state_builder.get_state(step_idx)
                        save_step_state(state)
                    else:
                        log.warning("No state builder attached; cannot save state")

                # Frame step triggers (edge)
                if self._input.down_event(self._forward_key):
                    self._render_frame = True
                    self._fps = self._base_fps
                elif self._input.down_event(self._backwards_key):
                    self._render_frame = True
                    self._frame_step_direction = False
                    self._fps = self._base_fps

                forward_pressed = self._input.is_pressed(self._forward_key)
                backward_pressed = self._input.is_pressed(self._backwards_key)

                # Multi-step modifier
                if self._input.is_pressed(self._multi_steps_key):
                    self._frame_step_size = 10
                else:
                    self._frame_step_size = 1

                # Fast modifier
                if (forward_pressed or backward_pressed) and self._input.is_pressed(self._fast_key):
                    self._fps = self._base_fps * 20
                    if backward_pressed:
                        self._frame_step_direction = False

                # Execute render frame if scheduled
                if self._render_frame:
                    current_frame_idx = self._renderer.get_current_frame_idx()
                    next_frame_idx = current_frame_idx + (
                        self._frame_step_size * (1 if self._frame_step_direction else -1)
                    )
                    next_frame_idx = max(next_frame_idx, 0)
                    self._renderer.render_frame(next_frame_idx)
                    self._last_render_time = time.time()
                    self._render_frame = False
                    self._frame_step_direction = True
                    self._frame_step_size = 1

                # Auto render timing
                if not self._paused and self._fps > 0:
                    fps_duration = 1 / self._fps
                    if time.time() - self._last_render_time > fps_duration:
                        self._render_frame = True

                # If paused disable auto FPS
                if self._paused:
                    self._fps = 0
                else:
                    self._fps = self._base_fps

                # Advance edge detection state if provider supports it
                end_frame = getattr(self._input, "end_frame", None)
                if callable(end_frame):
                    end_frame()
        finally:
            self.stop()
            if self._stop_flag is not None:
                self._stop_flag.value = True

    # Legacy internal methods removed; input handling now delegated to provider.

    def start(self) -> None:
        """Start render loop thread."""
        if self._running:
            return
        self._running = True
        self._loop_thread.start()

    def stop(self) -> None:
        """Stop the loop and cleanup."""
        if not self._running:
            return
        self._running = False
        log.debug("Stopping render loop")
        self._renderer.close()
        try:
            self._input.stop()
        except Exception:
            pass
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