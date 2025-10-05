from abc import ABC, abstractmethod
from typing import List
from snake_sim.environment.interfaces.loop_observer_interface import ILoopObserver
from snake_sim.environment.types import LoopStartData, LoopStepData, LoopStopData

class ILoopObservable:
    def __init__(self, *args, **kwargs):
        self._observers: List[ILoopObserver] = []
        self._did_notify_start = False
        self._did_notify_stop = False

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def stop(self):
        pass

    def add_observer(self, observer: ILoopObserver):
        self._observers.append(observer)

    def get_observers(self) -> List[ILoopObserver]:
        return self._observers

    @abstractmethod
    def _get_start_data(self) -> LoopStartData:
        pass

    @abstractmethod
    def _get_step_data(self) -> LoopStepData:
        pass

    @abstractmethod
    def _get_stop_data(self) -> LoopStopData:
        pass

    def _notify_start(self):
        if self._did_notify_start: return
        self._did_notify_start = True
        start_data = self._get_start_data()
        for observer in self._observers:
            observer.notify_start(start_data)

    def _notify_step(self):
        step_data = self._get_step_data()
        for observer in self._observers:
            observer.notify_step(step_data)

    def _notify_stop(self):
        if self._did_notify_stop: return
        self._did_notify_stop = True
        stop_data = self._get_stop_data()
        for observer in self._observers:
            observer.notify_stop(stop_data)

