import time

from typing import List
from threading import Condition

from snake_sim.environment.interfaces.loop_observer_interface import ILoopObserver


class WaitableObserver(ILoopObserver):
    """ Base class for loop data consumers. Just stores all data in memory. """
    def __init__(self):
        self._cond = Condition()
        self._is_cancelled = False
        self._has_started = False
        self._has_stopped = False
        self._current_step = None

    def _notify_cond(self):
        with self._cond:
            self._cond.notify_all()

    def notify_start(self, start_data):
        self._has_started = True
        self._notify_cond()

    def notify_step(self, step_data):
        self._current_step = step_data.step
        self._notify_cond()

    def notify_stop(self, stop_data):
        self._has_stopped = True
        self._notify_cond()

    def has_started(self) -> bool:
        return self._has_started

    def has_stopped(self) -> bool:
        return self._has_stopped

    def is_cancelled(self) -> bool:
        return self._is_cancelled

    def wait_for_step(self, step: int):
        with self._cond:
            self._cond.wait_for(
                lambda: self._current_step >= step or self._is_cancelled
            )

    def wait_until_finished(self):
        with self._cond:
            self._cond.wait_for(
                lambda: self._has_stopped or self._is_cancelled
            )
    
    def wait_until_started(self):
        with self._cond:
            self._cond.wait_for(
                lambda: self._has_started or self._is_cancelled
            )

    def cancel(self):
        self._is_cancelled = True
        self._notify_cond()
