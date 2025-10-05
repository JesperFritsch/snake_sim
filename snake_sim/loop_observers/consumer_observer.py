

from typing import List

from snake_sim.environment.types import (
    LoopStartData,
    LoopStepData,
    LoopStopData,
)
from snake_sim.environment.interfaces.loop_observer_interface import ILoopObserver


class ConsumerObserver(ILoopObserver):
    """ Base class for loop data consumers. Just stores all data in memory. """
    def __init__(self):
        self._start_data: LoopStartData = None
        self._steps: List[LoopStepData] = []
        self._stop_data: LoopStopData = None

    def notify_start(self, start_data: LoopStartData):
        self._start_data = start_data

    def notify_step(self, step_data: LoopStepData):
        if step_data.step != len(self._steps):
            raise RuntimeError(f"Received out of order step data. Expected step {len(self.steps)}, got {step_data.step}")
        self._steps.append(step_data)

    def notify_stop(self, stop_data: LoopStopData):
        self._stop_data = stop_data
