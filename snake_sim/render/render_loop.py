
import time
import logging

from pathlib import Path
from dataclasses import dataclass
from typing import Iterable
from threading import Thread, Event, Lock
from multiprocessing.sharedctypes import Synchronized
from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout

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
        self._render_lock: Lock = Lock()
        self._input_thread: Thread = Thread(target=self._input_loop)
        self._prompt_session: PromptSession = None
        self._frame_step_direction = True  # True for forward
        self._frame_step_size = 1
        self._base_fps = config.fps
        self._fps = self._base_fps
        self._paused = False  # stepping frames
        self._running = False  # loop running
        self._do_render_frame = False
        self._last_render_time = time.time()

        # Key identifiers (mapped inside provider)
        self._forward_key = "RIGHT"
        self._backwards_key = "LEFT"
        self._fast_key = "CTRL"
        self._multi_steps_key = "SHIFT"
        self._toggel_pause_key = "SPACE"
        self._save_state_key = "ENTER"
        self._quit_combo = ("CTRL", "C")
        self._go_end_combo = ("CTRL", "E")
        self._go_start_combo = ("CTRL", "S")
        self._go_mid_combo = ("CTRL", "M")

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
                try:
                    self._input = PygameInputProvider()
                    chosen = "pygame"
                except Exception:
                    chosen = None
                if chosen is None:
                    self._input = DummyInputProvider()
                    chosen = "dummy"
            log.debug("RenderLoop input provider selected: %s", chosen)

    def _input_loop(self):
        self._prompt_session = PromptSession()
        with patch_stdout():
            print("input thread started")
            while bool(command := self._prompt_session.prompt("> ")) and self._running_condition():
                try:
                    self._handle_command(command)
                except EOFError:
                    break

    def _render_step(self, step: int):
        with self._render_lock:
            self._renderer.render_step(step)

    def _render_frame(self, frame_idx: int):
        with self._render_lock:
            self._renderer.render_frame(frame_idx)

    def _handle_command(self, command: str):
        parts = command.split()
        if not parts:
            return
        cmd = parts[0]
        if cmd == "step":
            step = int(parts[1])
            self._render_step(step)
        elif cmd == "get":
            if parts[1] == "step":
                print(f"current step: {self._renderer.get_current_step_idx()}")
        else:
            print(f"invalid command {command}")

    def _running_condition(self):
        return (
                (not (self._stop_flag and self._stop_flag.value))
                and self._running
                and self._renderer.is_running()
                and not self._stop_event.is_set()
            )

    def _loop(self):
        self._paused = False
        try:
            while self._running_condition():
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
                    if self._input.is_pressed(keys=self._quit_combo):
                        log.info("Quit combo pressed; stopping render loop")
                        self._stop_event.set()

                # Toggle pause
                if self._input.down_event(self._toggel_pause_key):
                    self._paused = not self._paused

                # Save state
                if self._input.down_event(self._save_state_key):
                    if self._state_builder is not None:
                        step_idx = self._renderer.get_current_step_idx()
                        prev_state = self._state_builder.get_state(step_idx - 1)
                        curr_state = self._state_builder.get_state(step_idx)
                        next_state = self._state_builder.get_state(step_idx + 1)
                        save_step_state(prev_state, curr_state, next_state)
                    else:
                        log.warning("No state builder attached; cannot save state")

                if self._input.is_pressed(keys=self._go_end_combo):
                    self._renderer.render_last_frame()
                elif self._input.is_pressed(keys=self._go_start_combo):
                    self._renderer.render_first_frame()
                elif self._input.is_pressed(keys=self._go_mid_combo):
                    self._renderer.render_middle_frame()

                # Frame step triggers (edge)
                if self._input.down_event(self._forward_key):
                    self._do_render_frame = True
                    self._fps = self._base_fps
                elif self._input.down_event(self._backwards_key):
                    self._do_render_frame = True
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
                if self._do_render_frame:
                    current_frame_idx = self._renderer.get_current_map_idx()
                    next_frame_idx = current_frame_idx + (
                        self._frame_step_size * (1 if self._frame_step_direction else -1)
                    )
                    next_frame_idx = max(next_frame_idx, 0)
                    self._render_frame(next_frame_idx)
                    self._last_render_time = time.time()
                    self._do_render_frame = False
                    self._frame_step_direction = True
                    self._frame_step_size = 1

                # Auto render timing
                if not self._paused and self._fps > 0:
                    fps_duration = 1 / self._fps 
                    if time.time() - self._last_render_time > fps_duration:
                        self._do_render_frame = True

                if self._base_fps == -1:
                    self._do_render_frame = True

                # If paused disable auto FPS
                if self._paused and self._base_fps != -1:
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
        self._input_thread.start()

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
        try:
            self._prompt_session.app.exit()
        except:
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