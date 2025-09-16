
import logging
from pathlib import Path

from concurrent.futures import ThreadPoolExecutor, wait, as_completed
from typing import Tuple, List

from snake_sim.environment.interfaces.ISnakeUpdater import ISnakeUpdater
from snake_sim.environment.interfaces.snake_interface import ISnake
from snake_sim.environment.types import Coord, EnvData

log = logging.getLogger(Path(__file__).stem)

class ConcurrentUpdater(ISnakeUpdater):
    def __init__(self):
        self._executor: ThreadPoolExecutor = None
        self._snake_count = 0

    def get_decisions(self, snakes: List[ISnake], env_data: EnvData, timeout: int) -> dict[int, Coord]:
        futures = {self._executor.submit(snake.update, env_data): snake.get_id() for snake in snakes}
        decisions = {}
        for future in as_completed(futures, timeout=timeout):
            id = futures[future]
            decisions[id] = None
            try:
                decisions[id] = future.result()
            except TimeoutError:
                log.debug(f"Snake {id} timed out")
            except ConnectionError:
                log.debug(f"Error in snake {id}", exc_info=True)
        return decisions

    def close(self):
        self._executor.shutdown(wait=True)

    def register_snake(self, snake: ISnake):
        self._snake_count += 1
    
    def finalize(self):
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=self._snake_count)