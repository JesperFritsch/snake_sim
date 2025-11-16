
from typing import List, Set
import logging
from pathlib import Path

from uuid import uuid4
from snake_sim.environment.snake_updaters.concurrent_updater import ConcurrentUpdater
from snake_sim.snakes.shm_proxy_snake import SHMProxySnake
from snake_sim.environment.types import Coord, EnvStepData, EnvMetaData
from snake_sim.environment.shm_update import SharedMemoryWriter

log = logging.getLogger(Path(__file__).stem)

class SHMUpdater(ConcurrentUpdater):
    def __init__(self, ):
        super().__init__()
        self._shm_writer: SharedMemoryWriter = None
        self._managed_snakes: Set[SHMProxySnake] = set()

    def register_snake(self, snake: SHMProxySnake):
        self._confirm_snake_type(snake)
        snake.set_reader_id(self.snake_count)
        super().register_snake(snake)

    def unregister_snake(self, snake: SHMProxySnake):
        self._confirm_snake_type(snake)
        self._managed_snakes.remove(snake)
        super().unregister_snake(snake)

    def get_decisions(self, snakes: List[SHMProxySnake], env_step_data: EnvStepData, timeout: float) -> dict[int, Coord]:
        if any(not isinstance(snake, SHMProxySnake) for snake in snakes):
            raise TypeError("All snakes must be instances of SHMProxySnake.")
        self._shm_writer.write_frame(env_step_data.map.tobytes())
        return super().get_decisions(snakes, env_step_data, timeout)

    def close(self):
        if self._shm_writer is not None:
            self._shm_writer.cleanup()
        super().close()


    def finalize(self, env_meta_data: EnvMetaData):
        super().finalize(env_meta_data)
        self._create_shm_writer(env_meta_data)

    def _confirm_snake_type(self, snake: SHMProxySnake):
        if not isinstance(snake, SHMProxySnake):
            raise TypeError("All snakes must be instances of SHMProxySnake.")

    def _create_shm_writer(self, env_meta_data: EnvMetaData):
        if self._shm_writer is not None:
            return
        shm_name = str(uuid4())
        shm_reader_count = self.snake_count
        payload_size = env_meta_data.width * env_meta_data.height
        log.debug(f"Creating SharedMemoryWriter with name: {shm_name}, readers: {shm_reader_count}, payload_size: {payload_size}")
        self._shm_writer = SharedMemoryWriter.create(
            shm_name, shm_reader_count, payload_size
        )
        for snake in self._managed_snakes:
            snake.set_shm_name(shm_name)
