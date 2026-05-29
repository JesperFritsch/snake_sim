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

    def notify_decision(self, snake_id: int):
        """Fired when a single snake's decision for the current step has
        arrived from its updater. Lets out-of-process observers know which
        snakes are still pending mid-step (e.g. wall-clock budget
        enforcement when the sim hasn't finished the step yet). Default
        no-op."""
        pass

    def reset(self):
        """Optional reset method to clear internal state."""
        pass