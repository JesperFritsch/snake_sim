from abc import ABC, abstractmethod
from collections.abc import Iterator

from snake_sim.environment.types import (
    LoopStartData,
    LoopStepData,
    LoopStopData
)

class IRunFileHandler(ABC):

    @abstractmethod
    def __enter__(self) -> "IRunFileHandler":
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    @abstractmethod
    def write_start(self, start_data: LoopStartData):
        pass

    @abstractmethod
    def write_step(self, step_data: LoopStepData):
        pass

    @abstractmethod
    def write_stop(self, stop_data: LoopStopData):
        pass

    @abstractmethod
    def get_start_data(self) -> LoopStartData:
        pass

    @abstractmethod
    def iter_steps(self) -> Iterator[LoopStepData]:
        pass

    @abstractmethod
    def get_stop_data(self) -> LoopStopData:
        pass

    @classmethod
    def _get_file_format(cls) -> str:
        return cls.file_format

    def _default_filename(self) -> str:
        start_data = self.get_start_data()
        stop_data = self.get_stop_data()
        nr_snakes = len(start_data.env_init_data.start_positions)
        return f"run_{start_data.env_init_data.width}x{start_data.env_init_data.height}_{nr_snakes}snakes_{stop_data.final_step}steps{self._get_file_format()}"
