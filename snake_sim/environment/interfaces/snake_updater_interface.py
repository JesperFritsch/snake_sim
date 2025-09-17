import logging
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict

from snake_sim.environment.types import Coord, EnvData, EnvInitData
from snake_sim.environment.interfaces.snake_interface import ISnake

log = logging.getLogger(Path(__file__).stem)

class ISnakeUpdater(ABC):

    def __init__(self):
        self._snake_count = 0
    
    @abstractmethod
    def get_decisions(self, snakes: List[ISnake], env_data: EnvData, timeout: float) -> Dict[int, Coord]: # -> dict of snake id to direction
        pass

    def close(self):
        pass

    def register_snake(self, snake: ISnake):
        self._snake_count += 1

    def finalize(self, env_init_data: EnvInitData):
        log.debug(f"Finalizing updater {self.__class__.__name__} with {self._snake_count} snakes")

