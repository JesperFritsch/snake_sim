import random
import logging
import json
import random
import bisect

from typing import Dict, List, Tuple, Set
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib import resources

from snake_sim.environment.snake_processes import SnakeProcessManager
from snake_sim.environment.interfaces.snake_handler_interface import ISnakeHandler
from snake_sim.environment.interfaces.snake_interface import ISnake
from snake_sim.environment.types import Coord, EnvStepData, EnvMetaData

from snake_sim.environment.snake_updaters.inproc_updater import InprocUpdater
from snake_sim.environment.snake_updaters.shm_updater import SHMUpdater
from snake_sim.environment.snake_updaters.concurrent_updater import ConcurrentUpdater
from snake_sim.environment.interfaces.snake_updater_interface import ISnakeUpdater
from snake_sim.snakes.shm_proxy_snake import SHMProxySnake
from snake_sim.snakes.grpc_proxy_snake import GRPCProxySnake
from snake_sim.snakes.snake_base import SnakeBase
from snake_sim.rl.snakes.rl_snake_base import RLSnakeBase


with resources.open_text('snake_sim.config', 'default_config.json') as config_file:
    default_config = json.load(config_file)

log = logging.getLogger(Path(__file__).stem)


SNAKE_UPDATER_MAP = {
    SHMProxySnake: SHMUpdater,
    GRPCProxySnake: ConcurrentUpdater,
    SnakeBase: InprocUpdater,
    RLSnakeBase: ConcurrentUpdater,
}


class SnakeHandler(ISnakeHandler):
    def __init__(self):
        self._snakes: Dict[int, ISnake] = {}
        self._dead_snakes = set()
        self._executor = ThreadPoolExecutor(max_workers=len(SNAKE_UPDATER_MAP))
        self._updaters: Dict[type, ISnakeUpdater] = {}
        self._finalized = False

    def _get_updater(self, snake: ISnake) -> ISnakeUpdater:
        for snake_type, updater in SNAKE_UPDATER_MAP.items():
            if isinstance(snake, snake_type):
                updater = self._updaters.setdefault(snake_type, updater())
                return updater

    def get_next_snake_id(self) -> int:
        return len(self._snakes)

    def get_snakes(self) -> Dict[int, ISnake]:
        return self._snakes.copy()

    def kill_snake(self, id):
        log.debug(f"Killing snake with id {id}")
        # SnakeProcessManager().kill_snake_process(id) # Dont kill the process for RL training
        snake = self._snakes.get(id)
        if snake is not None:
            updater = self._get_updater(snake)
            updater.unregister_snake(snake)
        return self._dead_snakes.add(id)

    def _split_batch_by_updater(self, batch_data: Dict[int, EnvStepData]) -> Dict[ISnakeUpdater, Tuple[List[ISnake], EnvStepData]]:
        updater_batches: Dict[ISnakeUpdater, Tuple[List[ISnake], EnvStepData]] = {}
        for id, env_step_data in batch_data.items():
            snake = self._snakes[id]
            updater = self._get_updater(snake)
            updater_batches.setdefault(updater, ([], env_step_data))[0].append(snake)
        return updater_batches

    def _gather_decisions(self, updater_batches: Dict[ISnakeUpdater, Tuple[List[ISnake], EnvStepData]]) -> Dict[int, Coord]:
        decisions = {}
        timeout = default_config["decision_timeout_ms"] / 1000
        futures = [
            self._executor.submit(updater.get_decisions, snakes, env_step_data, timeout)
            for updater, (snakes, env_step_data) in updater_batches.items()
        ]
        for future in as_completed(futures):
            decisions.update(future.result())
        return decisions

    def get_decisions(self, batch_data: Dict[int, EnvStepData]) -> Dict[int, Coord]:
        updater_batches = self._split_batch_by_updater(batch_data)
        return self._gather_decisions(updater_batches)

    def add_snake(self, snake: ISnake):
        if snake in self._snakes.values():
            return
        snake_id = self.get_next_snake_id()
        self._snakes[snake_id] = snake
        updater = self._get_updater(snake)
        updater.register_snake(snake)

    def _create_in_range_map(self, position_data: Dict[int, Coord]) -> Dict[int, Set[int]]: # Dict[id, List[id]]
        """ Creates a map of snakes that are in range of each other, meaning they can end up on the same tile in one move """
        in_range_map = {}
        for pos_item in position_data.items():
            s_id, pos = pos_item
            in_range_map[s_id] = {s_id2 for s_id2, pos2 in position_data.items() if pos.distance(pos2) <= 2 and s_id != s_id2}
        return in_range_map

    def get_batch_order(self, position_data: Dict[int, Coord]) -> List[List[int]]:
        in_range_map = self._create_in_range_map(position_data)
        ids_to_batch = list(position_data.keys())
        batches: List[List[int]] = []
        while ids_to_batch:
            current_id = ids_to_batch.pop()
            in_range_to_current = in_range_map[current_id]
            for batch in batches:
                if not any(batch_id in in_range_to_current for batch_id in batch):
                    batch.append(current_id)
                    break
            else:
                batches.append([current_id])
        random.shuffle(batches)
        return batches

    def get_update_order(self) -> list:
        ids = [id for id in self._snakes.keys() if id not in self._dead_snakes]
        random.shuffle(ids)
        return ids

    def finalize(self, env_meta_data: EnvMetaData):
        if self._finalized:
            return
        self._finalized = True
        log.debug(f"Finalizing these updaters ({', '.join([type(u).__name__ for u in self._updaters.values()])})")
        for updater in self._updaters.values():
            updater.finalize(env_meta_data)

    def reset(self, start_positions: Dict[int, Coord]):
        log.debug("Resetting SnakeHandler")
        for snake_id, snake in self._snakes.items():
            snake.reset()
            snake.set_start_position(start_positions[snake_id])
            updater = self._get_updater(snake)
            updater.register_snake(snake)
        self._dead_snakes.clear()

    def close(self):
        log.debug("Closing SnakeHandler")
        for updater in self._updaters.values():
            updater.close()
        self._executor.shutdown(wait=True)
