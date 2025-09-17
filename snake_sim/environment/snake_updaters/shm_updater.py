
from typing import Tuple, List, Dict

from uuid import uuid4
from snake_sim.environment.snake_updaters.concurrent_updater import ConcurrentUpdater
from snake_sim.snakes.shm_proxy_snake import SHMProxySnake
from snake_sim.environment.types import Coord, EnvData
from snake_sim.environment.shm_update import SharedMemoryWriter


class SHMUpdater(ConcurrentUpdater):
    def __init__(self, ):
        super().__init__()
        self._shm_writer: SharedMemoryWriter = None
        self._added_snakes = set()

    def register_snake(self, snake: SHMProxySnake):
        if snake_id in self._added_snakes:
            raise ValueError(f"Snake with ID {snake_id} is already registered.")
        snake.set_reader_id(self._snake_count)
        self._added_snakes.insert(snake)
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
        shm_name = str(uuid4())
        shm_reader_count = self._snake_count
        payload_size = #Size of map
        raise RuntimeError("SHM updater not complete")
        self._shm_writer = SharedMemoryWriter.create()
        return super().finalize()
    
