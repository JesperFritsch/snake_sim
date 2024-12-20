from abc import ABC, abstractmethod


class ILoopObserver(ABC):

    @abstractmethod
    def notify_start(self):
        pass

    @abstractmethod
    def notify_step(self, loop_data):
        pass

    @abstractmethod
    def notify_end(self):
        pass