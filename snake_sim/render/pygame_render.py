

import time
import numpy as np
import logging
import pygame

from pathlib import Path
from threading import Thread, Event, Lock

from snake_sim.render.interfaces.renderer_interface import IRenderer
from snake_sim.loop_observers.map_builder_observer import MapBuilderObserver, NoMoreSteps, CurrentIsFirst
from snake_sim.render.utils import create_color_map

log = logging.getLogger(Path(__file__).stem)


class PygameRenderer(IRenderer):
    """ A simple terminal renderer that prints the state to the console. """
    def __init__(self, map_builder: MapBuilderObserver, screen_h: int = 1000, screen_w: int = 1000):
        super().__init__()
        self._map_builder = map_builder
        self._screen_h = screen_h
        self._screen_w = screen_w
        self._wait_thread = Thread(target=self._finish_init, daemon=True)
        self._wait_thread.start()
        self._env_meta_data = None
        self._color_map = None
        self._free_value = None
        self._food_value = None
        self._blocked_value = None
        self._init_finished = False
        self._flip_event: Event = Event()
        self._close_event: Event = Event()
        self._pygame_thread = Thread(target=self._pygame_loop)
        self._pygame_thread.start()

    def _finish_init(self):
        while self._map_builder._start_data is None:
            time.sleep(0.005)
        self._env_meta_data = self._map_builder._start_data.env_meta_data
        self._color_map = create_color_map(self._env_meta_data.snake_values)
        self._free_value = self._env_meta_data.free_value
        self._food_value = self._env_meta_data.food_value
        self._blocked_value = self._env_meta_data.blocked_value
        self._init_finished = True

    def is_init_finished(self):
        return self._init_finished

    def _render_frame(self, frame: np.ndarray):
        if not self.is_init_finished():
            log.debug("Skipping render frame; init not finished")
            return
        unique_ids, dense_labels = np.unique(frame, return_inverse=True)
        lut = np.array([self._color_map[i] for i in unique_ids], dtype=np.uint8)
        color_frame = lut[dense_labels].reshape(*frame.shape, 3)
        self.draw_frame(color_frame)

    def render_step(self, step_idx: int):
        try:
            frame = self._map_builder.get_map_for_step(step_idx)
            self._render_frame(frame)
        except (StopIteration, NoMoreSteps, CurrentIsFirst):
            pass

    def render_frame(self, frame_idx: int):
        try:
            frame = self._map_builder.get_map(frame_idx)
            self._render_frame(frame)
        except (StopIteration, NoMoreSteps, CurrentIsFirst):
            pass

    def get_current_map_idx(self):
        return self._map_builder.get_current_map_idx()

    def get_current_step_idx(self):
        return self._map_builder.get_current_step_idx()

    def is_running(self) -> bool:
        return not self._close_event.is_set()

    def _pygame_loop(self):
        # Continuously poll events so the OS considers the window responsive.
        # Run in a daemon thread; keep the loop light-weight so it doesn't hog CPU.
        try:
            self._screen = pygame.display.set_mode((self._screen_w, self._screen_h), 0, 32)
            self._surface = pygame.Surface(self._screen.get_size()).convert()
            while not self._close_event.is_set():
                if self._flip_event.is_set():
                    self._flip_event.clear()
                    pygame.display.flip()
                try:
                    # Pump internal events before flipping to keep the window responsive.
                    pygame.event.pump()
                except Exception:
                    # If the display has been closed or pump fails, ignore here and allow quit handling elsewhere
                    pass
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise KeyboardInterrupt
                # Yield to avoid busy loop
                time.sleep(0.01)
        except KeyboardInterrupt:
            pass
        finally:
            try:
                # we can not quit pygame while we are drawing a frame
                self._close_event.set()
                log.debug("Quitting pygame")
                pygame.quit()
            except:
                log.error("Failed to quit pygame")
                log.debug("TRACE: ", exc_info=True)

    def draw_frame(self, frame_buffer):
        # we can not draw frames while we are shutting down
        if self._close_event.is_set():
            return
        # frame buffer is expected to be of shape (h, w, 3) with rgb values
        frame_buffer = np.rot90(np.fliplr(frame_buffer))
        buffer_surface = pygame.surfarray.make_surface(frame_buffer)
        scaled_surface = pygame.transform.scale(buffer_surface, (self._screen_w, self._screen_h))
        self._screen.blit(scaled_surface, (0, 0))
        self._flip_event.set()

    def close(self):
        self._close_event.set()
        self._pygame_thread.join()