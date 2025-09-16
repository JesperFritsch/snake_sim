
from typing import Tuple, List, Dict

from snake_sim.environment.snake_updaters.concurrent_updater import ConcurrentUpdater
from snake_sim.snakes.shm_proxy_snake import SHMProxySnake
from snake_sim.environment.types import Coord, EnvData
from snake_sim.environment.shm_update import SharedMemoryWriter


class SHMUpdater(ConcurrentUpdater):
    def __init__(self, ):
        super().__init__()
        self._shm_writer: SharedMemoryWriter = None
        self._snake_reader_map: Dict[int, int] = {}  # snake id to reader id
        self._next_reader_id = 0

    def register_snake(self, snake: SHMProxySnake):
        if self._shm_writer is None:
            raise RuntimeError("SHMUpdater not initialized with SharedMemoryWriter.")
        snake_id = snake.get_id()
        if snake_id is None:
            raise ValueError("Snake ID is not set. Cannot register snake.")
        if snake_id in self._snake_reader_map:
            raise ValueError(f"Snake with ID {snake_id} is already registered.")
        reader_id = self._next_reader_id
        self._next_reader_id += 1
        self._snake_reader_map[snake_id] = reader_id
        snake.set_reader_id(reader_id)
        super().register_snake(snake)

    def get_decisions(self, snakes: List[SHMProxySnake], env_data: EnvData) -> dict[int, Coord]:
        if any(not isinstance(snake, SHMProxySnake) for snake in snakes):
            raise TypeError("All snakes must be instances of SHMProxySnake.")
        self._shm_writer.write_frame(env_data.map)
        return super().get_decisions(snakes, env_data)

    def close(self):
        self._shm_writer.close()
        super().close()

    def finalize(self):
        return super().finalize()
    