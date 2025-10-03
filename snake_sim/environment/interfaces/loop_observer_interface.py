from abc import ABC, abstractmethod
from snake_sim.environment.types import LoopStartData, LoopStepData, LoopStopData

class ILoopObserver(ABC):

    @abstractmethod
    def notify_start(self, start_data: LoopStartData):
        pass

    @abstractmethod
    def notify_step(self, step_data: LoopStepData):
        pass

    @abstractmethod
    def notify_stop(self, stop_data: LoopStopData):
        pass