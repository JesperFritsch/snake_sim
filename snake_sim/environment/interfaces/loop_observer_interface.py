from abc import ABC, abstractmethod
from snake_sim.environment.types import LoopStepData

class ILoopObserver(ABC):

    @abstractmethod
    def notify_start(self):
        pass

    @abstractmethod
    def notify_step(self, step_data: LoopStepData):
        pass

    @abstractmethod
    def notify_end(self):
        pass