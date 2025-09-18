
import logging
from pathlib import Path

from concurrent.futures import ThreadPoolExecutor, wait, as_completed
from typing import Tuple, List

from snake_sim.environment.interfaces.snake_updater_interface import ISnakeUpdater
from snake_sim.environment.interfaces.snake_interface import ISnake
from snake_sim.environment.types import Coord, EnvData, EnvInitData

log = logging.getLogger(Path(__file__).stem)

class ConcurrentUpdater(ISnakeUpdater):
    def __init__(self):
        super().__init__()
        self._executor: ThreadPoolExecutor = None

    def get_decisions(self, snakes: List[ISnake], env_data: EnvData, timeout: float) -> dict[int, Coord]:
        futures = {self._executor.submit(snake.update, env_data): snake.get_id() for snake in snakes}
        decisions = {}
        try:
            for future in as_completed(futures, timeout=timeout):
                id = futures[future]
                decisions[id] = None
                try:
                    decisions[id] = future.result()
                except ConnectionError:
                    log.debug(f"Snake with id {id} disconnected.")
        except TimeoutError:
            pass
        return decisions

    def close(self):
        super().close()
        self._executor.shutdown(wait=True)

    def finalize(self, env_init_data: EnvInitData):
        super().finalize(env_init_data)
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=self._snake_count)
