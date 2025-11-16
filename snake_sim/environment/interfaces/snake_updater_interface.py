import logging
import sys
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Set

from snake_sim.environment.types import Coord, EnvStepData, EnvMetaData
from snake_sim.environment.interfaces.snake_interface import ISnake

log = logging.getLogger(Path(__file__).stem)

class ISnakeUpdater(ABC):

    def __init__(self):
        self._managed_snakes: Set[ISnake] = set()
        self._finalized = False

    @abstractmethod
    def get_decisions(self, snakes: List[ISnake], env_step_data: EnvStepData, timeout: float) -> Dict[int, Coord]: # -> dict of snake id to direction
        pass

    @property
    def snake_count(self) -> int:
        return len(self._managed_snakes)

    def close(self):
        log.debug(f"Closing updater {self.__class__.__name__} {id(self)}")
        sys.stdout.flush()

    def register_snake(self, snake: ISnake):
        self._managed_snakes.add(snake)

    def unregister_snake(self, snake: ISnake):
        self._managed_snakes.discard(snake)

    def finalize(self, env_meta_data: EnvMetaData):
        if self._finalized:
            return
        self._finalized = True
        log.debug(f"Finalizing updater {self.__class__.__name__} {id(self)} with {self.snake_count} snakes")
    
