
import logging
from pathlib import Path

from concurrent.futures import ThreadPoolExecutor, wait, as_completed
from concurrent.futures import TimeoutError as concurrentTimeoutError
from typing import Tuple, List

from snake_sim.environment.interfaces.snake_updater_interface import ISnakeUpdater
from snake_sim.environment.interfaces.snake_interface import ISnake
from snake_sim.environment.types import Coord, EnvStepData, EnvMetaData

log = logging.getLogger(Path(__file__).stem)

class ConcurrentUpdater(ISnakeUpdater):
    def __init__(self):
        super().__init__()
        self._executor: ThreadPoolExecutor = None

    def get_decisions(self, snakes: List[ISnake], env_step_data: EnvStepData, timeout: float) -> dict[int, Coord]:
        futures = {self._executor.submit(snake.update, env_step_data): snake.get_id() for snake in snakes}
        decisions = {snake.get_id(): None for snake in snakes}
        try:
            for future in as_completed(futures, timeout=timeout):
                id = futures[future]
                try:
                    decisions[id] = future.result()
                except ConnectionError:
                    log.debug(f"Snake with id {id} disconnected.")
        except concurrentTimeoutError:
            pass
        return decisions

    def close(self):
        super().close()
        if self._executor is not None:
            self._executor.shutdown(wait=True)

    def finalize(self, env_meta_data: EnvMetaData):
        super().finalize(env_meta_data)
        if self._executor is None:
            log.debug(f"Creating ThreadPoolExecutor with {self.snake_count} workers")
            self._executor = ThreadPoolExecutor(max_workers=self.snake_count)
