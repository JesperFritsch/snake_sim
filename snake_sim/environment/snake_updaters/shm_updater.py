
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
        self._managed_snakes: Set[SHMProxySnake] = set()

    def register_snake(self, snake: SHMProxySnake):
        self._confirm_snake_type(snake)
        if snake in self._managed_snakes:
            raise ValueError(f"Snake with ID {snake.get_id()} is already registered.")
        snake.set_reader_id(self._snake_count)
        self._managed_snakes.add(snake)
        super().register_snake(snake)
    
    def unregister_snake(self, snake: SHMProxySnake):
        self._confirm_snake_type(snake)
        if snake not in self._managed_snakes:
            raise ValueError(f"Snake with ID {snake.get_id()} is not registered.")
        self._managed_snakes.remove(snake)
        super().unregister_snake(snake)

    def get_decisions(self, snakes: List[SHMProxySnake], env_data: EnvData, timeout: float) -> dict[int, Coord]:
        if any(not isinstance(snake, SHMProxySnake) for snake in snakes):
            raise TypeError("All snakes must be instances of SHMProxySnake.")
        self._shm_writer.write_frame(env_data.map)
        return super().get_decisions(snakes, env_data, timeout)

    def close(self):
        if self._shm_writer is not None:
            self._shm_writer.cleanup()
        super().close()


    def finalize(self, env_init_data: EnvInitData):
        super().finalize(env_init_data)
        self._create_shm_writer(env_init_data)
    
    def _confirm_snake_type(self, snake: SHMProxySnake):
        if not isinstance(snake, SHMProxySnake):
            raise TypeError("All snakes must be instances of SHMProxySnake.")

    def _create_shm_writer(self, env_init_data: EnvInitData):
        if self._shm_writer is not None:
            return
        shm_name = str(uuid4())
        shm_reader_count = self._snake_count
        payload_size = env_init_data.width * env_init_data.height
        log.debug(f"Creating SharedMemoryWriter with name: {shm_name}, readers: {shm_reader_count}, payload_size: {payload_size}")
        self._shm_writer = SharedMemoryWriter.create(
            shm_name, shm_reader_count, payload_size
        )
        for snake in self._managed_snakes:
            snake.set_shm_name(shm_name)
