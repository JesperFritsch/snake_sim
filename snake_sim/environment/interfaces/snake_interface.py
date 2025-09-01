
from abc import ABC, abstractmethod
from snake_sim.environment.types import Coord, EnvInitData, EnvData
from typing import Tuple


class ISnake(ABC):

    @abstractmethod
    def set_id(self, id: int):
        pass

    @abstractmethod
    def set_start_length(self, start_length: int):
        pass

    @abstractmethod
    def set_start_position(self, start_position: Coord):
        pass

    @abstractmethod
    def set_init_data(self, env_data: EnvInitData):
        pass

    @abstractmethod
    def update(self, env_data: EnvData) -> Coord: # -> (int, int) as direction (1, 0) for right (-1, 0) for left
        pass

