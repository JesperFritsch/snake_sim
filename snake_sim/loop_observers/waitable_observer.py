import time

from typing import List

from snake_sim.environment.interfaces.loop_observer_interface import ILoopObserver


class WaitableObserver(ILoopObserver):
    """ Base class for loop data consumers. Just stores all data in memory. """
    def __init__(self):
        self._wait_step = None
        self._has_started = False
        self._has_stopped = False
        self._current_step = None
        self._is_waiting = False

    def notify_start(self, start_data):
        self._has_started = True

    def notify_step(self, step_data):
        self._current_step = step_data.step
        if self._wait_step is not None and self._current_step >= self._wait_step:
            self._is_waiting = False

    def notify_stop(self, stop_data):
        self._has_stopped = True
        self._is_waiting = False

    def wait_for_step(self, step: int):
        self._wait_step = step
        self._is_waiting = True
        while self._is_waiting and not self._has_stopped:
            time.sleep(0.01)

    def has_started(self) -> bool:
        return self._has_started

    def wait_until_finished(self):
        while self._has_started and not self._has_stopped:
            time.sleep(0.01)
