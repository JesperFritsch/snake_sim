import logging
from pathlib import Path
from abc import ABC, abstractmethod
from typing import List
from snake_sim.environment.interfaces.loop_observer_interface import ILoopObserver
from snake_sim.environment.types import LoopStartData, LoopStepData, LoopStopData, LoopDecisionData

log = logging.getLogger(Path(__file__).stem)

class ILoopObservable:
    def __init__(self, *args, **kwargs):
        self._observers: List[ILoopObserver] = []
        self._did_notify_start = False
        self._did_notify_stop = False

    def is_done(self) -> bool:
        return self._did_notify_stop

    def is_running(self) -> bool:
        return self._did_notify_start and not self.is_done()

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def stop(self):
        pass

    def close(self):
        for observer in self._observers:
            if hasattr(observer, 'close'):
                observer.close()

    def add_observer(self, observer: ILoopObserver):
        self._observers.append(observer)

    def get_observers(self) -> List[ILoopObserver]:
        return self._observers

    def _notify_start(self, start_data: LoopStartData):
        if self._did_notify_start: return
        log.debug(f"{self.__class__.__name__} notifying observers of loop start")
        self._did_notify_start = True
        for observer in self._observers:
            observer.notify_start(start_data)

    def _notify_step(self, step_data: LoopStepData):
        for observer in self._observers:
            observer.notify_step(step_data)

    def _notify_decision(self, decision_data: LoopDecisionData):
        print(f"Notifying observers of decision: {decision_data}")
        for observer in self._observers:
            observer.notify_decision(decision_data)

    def _notify_stop(self, stop_data: LoopStopData):
        if self._did_notify_stop: return
        log.debug(f"{self.__class__.__name__} notifying observers of loop stop")
        self._did_notify_stop = True
        for observer in self._observers:
            observer.notify_stop(stop_data)

