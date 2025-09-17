
from typing import List, Set
import logging
from pathlib import Path

from uuid import uuid4
from snake_sim.environment.snake_updaters.concurrent_updater import ConcurrentUpdater
from snake_sim.snakes.shm_proxy_snake import SHMProxySnake
from snake_sim.environment.types import Coord, EnvData, EnvInitData
from snake_sim.environment.shm_update import SharedMemoryWriter

log = logging.getLogger(Path(__file__).stem)

class SHMUpdater(ConcurrentUpdater):
    def __init__(self, ):
        super().__init__()
        self._shm_writer: SharedMemoryWriter = None
        self._added_snakes: Set[SHMProxySnake] = set()

    def register_snake(self, snake: SHMProxySnake):
        if snake in self._added_snakes:
            raise ValueError(f"Snake with ID {snake.get_id()} is already registered.")
        snake.set_reader_id(self._snake_count)
        self._added_snakes.add(snake)
        super().register_snake(snake)

    def get_decisions(self, snakes: List[SHMProxySnake], env_data: EnvData, timeout: float) -> dict[int, Coord]:
        if any(not isinstance(snake, SHMProxySnake) for snake in snakes):
            raise TypeError("All snakes must be instances of SHMProxySnake.")
        self._shm_writer.write_frame(env_data.map)
        return super().get_decisions(snakes, env_data, timeout)

    def close(self):
        log.debug("Closing SHMUpdater and cleaning up SharedMemoryWriter")
        self._shm_writer.cleanup()
        super().close()

    def finalize(self, env_init_data: EnvInitData):
        super().finalize(env_init_data)
        shm_name = str(uuid4())
        shm_reader_count = self._snake_count
        payload_size = env_init_data.width * env_init_data.height
        log.debug(f"Creating SharedMemoryWriter with name: {shm_name}, readers: {shm_reader_count}, payload_size: {payload_size}")
        self._shm_writer = SharedMemoryWriter.create(
            shm_name, shm_reader_count, payload_size
        )
        for snake in self._added_snakes:
            snake.set_shm_name(shm_name)
