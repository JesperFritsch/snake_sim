
from abc import ABC, abstractmethod
from snake_sim.utils import Coord


class ISnake(ABC):
    @abstractmethod
    def __init__(self, id: int, start_length: int):
        pass

    @abstractmethod
    def get_id(self) -> int:
        pass

    @abstractmethod
    def get_length(self) -> int:
        pass

    @abstractmethod
    def set_start_position(self, start_position: Coord):
        pass

    @abstractmethod
    def set_init_data(self, env_data: dict):
        pass

    @abstractmethod
    def update(self, env_data: dict) -> Coord: # -> (int, int) as direction (1, 0) for right (-1, 0) for left
        pass