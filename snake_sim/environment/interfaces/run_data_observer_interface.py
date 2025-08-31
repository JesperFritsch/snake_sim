from abc import ABC, abstractmethod

from snake_sim.run_data.run_data import StepData, RunData

class IRunDataObserver(ABC):

    @abstractmethod
    def notify_start(self, metadata: dict):
        pass

    @abstractmethod
    def notify_step(self, step_data: StepData):
        pass

    @abstractmethod
    def notify_end(self, run_data: RunData):
        pass
