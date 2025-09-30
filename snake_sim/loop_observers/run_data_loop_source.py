import logging
from pathlib import Path
from typing import Set

from snake_sim.environment.interfaces.main_loop_interface import ILoopObserver
from snake_sim.loop_observables.main_loop import LoopStartData, LoopStepData, LoopStopData
from snake_sim.data_adapters.run_data_adapter import RunDataAdapter
from snake_sim.environment.interfaces.run_data_observer_interface import IRunDataObserver

log = logging.getLogger(Path(__file__).stem)

class RunDataSource(ILoopObserver):
    """ This is a class that is used to observe a main loop and convert the data to RunData and StepData.
    The purpose is to observe the main loop observer and store the data in a run data object. this can be used for both storing the run to file and
    for transferring the data to pygame for visualization. """
    def __init__(self):
        self._adapter = None
        self._observers: Set[IRunDataObserver] = set()
        self._has_started = False

    def notify_start(self, start_data: LoopStartData):
        if not isinstance(self._adapter, RunDataAdapter):
            raise ValueError('Adapter not set')
        self._has_started = True
        for observer in self._observers:
            observer.notify_start(self._adapter.get_metadata())

    def notify_step(self, loop_step: LoopStepData):
        if not self._has_started:
            log.warning('Not started yet')
        step_data = self._adapter.loop_step_data_to_step_data(loop_step)
        for observer in self._observers:
            observer.notify_step(step_data)

    def notify_end(self, stop_data: LoopStopData):
        if not self._has_started:
            return
        for observer in self._observers:
            observer.notify_end(self._adapter.get_run_data())

    def set_adapter(self, adapter: RunDataAdapter):
        if not isinstance(adapter, RunDataAdapter):
            raise ValueError('adapter must be of type RunDataAdapter')
        self._adapter = adapter

    def add_observer(self, observer: IRunDataObserver):
        if not isinstance(observer, IRunDataObserver):
            raise ValueError('observer must be of type IRunDataObserver')
        self._observers.add(observer)

    def get_observers(self):
        return list(self._observers)

    def remove_observer(self, observer: IRunDataObserver):
        if not isinstance(observer, IRunDataObserver):
            raise ValueError('observer must be of type IRunDataObserver')
        self._observers.remove(observer)
