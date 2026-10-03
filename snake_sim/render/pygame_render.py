import time
import numpy as np
import logging
import pygame

from pathlib import Path
from threading import Thread, Event

from snake_sim.render.base_frame_renderer import BaseFrameRenderer
from snake_sim.render.interfaces.frame_producer_interface import IFrameProducer

log = logging.getLogger(Path(__file__).stem)


class PygameRenderer(BaseFrameRenderer):
    """ Blits finished frame buffers into a pygame window, scaled to fit it. """
    def __init__(self, frame_producer: IFrameProducer, max_screen_size: int = 1000):
        super().__init__(frame_producer)
        self._max_screen_size = max_screen_size
        self._screen_h = 100
        self._screen_w = 100
        self._init_finished = False
        self._loop_started = False
        self._flip_event: Event = Event()
        self._close_event: Event = Event()
        self._pygame_thread = Thread(target=self._pygame_loop)
        # Started last; _finish_init touches the attributes set above.
        self._wait_thread = Thread(target=self._finish_init, daemon=True)
        self._wait_thread.start()

    def _find_correct_screen_size(self, frame_width: int, frame_height: int):
        aspect_ratio = frame_width / frame_height
        if aspect_ratio >= 1:
            self._screen_w = self._max_screen_size
            self._screen_h = int(self._max_screen_size / aspect_ratio)
        else:
            self._screen_h = self._max_screen_size
            self._screen_w = int(self._max_screen_size * aspect_ratio)

    def _finish_init(self):
        self._producer.wait_until_ready()
        self._find_correct_screen_size(*self._producer.size)
        self._pygame_thread.start()
        self._init_finished = True

    def is_init_finished(self):
        return self._init_finished and self._loop_started

    def _render_at(self, frame_idx: int):
        if not self.is_init_finished():
            log.debug("Skipping render frame; init not finished")
            return
        self.draw_frame(self._producer.get_frame(frame_idx).pixels)

    def is_running(self) -> bool:
        return not self._close_event.is_set()

    def _pygame_loop(self):
        # Continuously poll events so the OS considers the window responsive.
        # Run in a daemon thread; keep the loop light-weight so it doesn't hog CPU.
        try:
            self._screen = pygame.display.set_mode((self._screen_w, self._screen_h), 0, 32)
            self._surface = pygame.Surface(self._screen.get_size()).convert()
            self._loop_started = True
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
