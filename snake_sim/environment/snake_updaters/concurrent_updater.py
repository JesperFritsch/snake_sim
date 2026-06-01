
import logging
import time
from pathlib import Path

from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as concurrentTimeoutError
from typing import Callable, Tuple, List

from snake_sim.environment.interfaces.snake_updater_interface import ISnakeUpdater
from snake_sim.environment.interfaces.snake_interface import ISnake
from snake_sim.environment.types import Coord, EnvStepData, EnvMetaData, LoopDecisionData

log = logging.getLogger(Path(__file__).stem)

class ConcurrentUpdater(ISnakeUpdater):
    def __init__(self):
        super().__init__()
        self._executor: ThreadPoolExecutor = None

    def get_decisions(
        self,
        snakes: List[ISnake],
        env_step_data: EnvStepData,
        timeout_s: float | None,
        on_response: Callable[[int, int], None],
    ) -> dict[int, Coord]:
        decisions = {snake.get_id(): None for snake in snakes}
        completed_futures = set()
        start_time = time.monotonic_ns()
        futures = {self._executor.submit(snake.update, env_step_data): snake.get_id() for snake in snakes}
        try:
            for future in as_completed(futures, timeout=timeout_s):
                decision_wall_time = time.monotonic_ns() - start_time
                id = futures[future]
                completed_futures.add(future)
                try:
                    decisions[id] = future.result()
                except ConnectionError:
                    log.debug(f"Snake with id {id} disconnected.")
                try:
                    on_response(id, decision_wall_time)
                except Exception:
                    log.exception("on_response callback failed for snake %s", id)
        except concurrentTimeoutError:
            pass
        # Snakes whose futures did not complete in time
        timed_out_ids = [futures[f] for f in set(futures) - completed_futures]
        if timed_out_ids:
            log.debug(f"Snakes with ids {timed_out_ids} timed out.")
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
