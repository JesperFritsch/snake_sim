import random
import logging
import json
import random
import time

from typing import Dict, List
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, wait, as_completed
from importlib import resources

from snake_sim.environment.snake_processes import SnakeProcessPool
from snake_sim.environment.interfaces.snake_handler_interface import ISnakeHandler
from snake_sim.snakes.snake import ISnake
from snake_sim.snakes.remote_snake import RemoteSnake
from snake_sim.environment.snake_env import EnvData
from snake_sim.utils import Coord

with resources.open_text('snake_sim.config', 'default_config.json') as config_file:
    default_config = json.load(config_file)

log = logging.getLogger(Path(__file__).stem)


class SnakeHandler(ISnakeHandler):
    def __init__(self):
        self._snakes: Dict[int, ISnake] = {}
        self._dead_snakes = set()
        self._executor = None

    def _init_executor(self):
        nr_snakes = len(self._snakes)
        # max workers is the number of remote snakes, because a batch will never be bigger than the number of remote snakes
        self._executor = ThreadPoolExecutor(max_workers=nr_snakes * 2)

    def get_snakes(self) -> Dict[int, ISnake]:
        return self._snakes.copy()

    def kill_snake(self, id):
        SnakeProcessPool().kill_snake_process(id)
        return self._dead_snakes.add(id)

    def _process_batch_sync(self, batch_data: Dict[int, EnvData]) -> Dict[int, Coord]:
        decisions = {}
        for id, env_data in batch_data.items():
            decisions[id] = self.get_decision(id, env_data)
        return decisions

    def _process_batch_concurrent(self, batch_data: Dict[int, EnvData]) -> Dict[int, Coord]:
        if not self._executor:
            self._init_executor()
        # print("Processing batch: ", list(batch_data.keys()))
        futures = {self._executor.submit(self.get_decision, id, env_data): id for id, env_data in batch_data.items()}
        decisions = {}
        for future in as_completed(futures, timeout=default_config["decision_timeout_ms"] / 1000):
            id = futures[future]
            try:
                decisions[id] = future.result()
            except TimeoutError:
                print(f"Snake {id} timed out")
                self.kill_snake(id)
                decisions[id] = None
            except Exception as e:
                print(f"Error in snake {id}", exc_info=True)
                self.kill_snake(id)
                decisions[id] = None
        return decisions

    def get_decisions(self, batch_data: Dict[int, EnvData]) -> Dict[int, Coord]:
        if any(isinstance(self._snakes[snake_id], RemoteSnake) for snake_id in batch_data.keys()):
            return self._process_batch_concurrent(batch_data)
        return self._process_batch_sync(batch_data)

    def get_decision(self, id, env_data: EnvData) -> Coord:
        snake = self._snakes[id]
        decision = snake.update(env_data)
        try:
            decision_coord = Coord(*decision)
        except:
            return
        return decision_coord

    def add_snake(self, id, snake: ISnake):
        self._snakes[id] = snake

    def _create_in_range_map(self, position_data: Dict[int, Coord]) -> Dict[int, List[int]]: # Dict[id, List[id]]
        """ Creates a map of snakes that are in range of each other, meaning they can end up on the same tile in one move """
        in_range_map = {}
        for id, pos in position_data.items():
            in_range_map[id] = [id2 for id2, pos2 in position_data.items() if pos.distance(pos2) <= 2 and id != id2]
        return in_range_map

    def get_batch_order(self, position_data: Dict[int, Coord]) -> List[List[int]]:
        in_range_map = self._create_in_range_map(position_data)
        def can_go_in_batch(batch, id):
            for batch_id in batch:
                if id in in_range_map[batch_id]:
                    return False
            return True
        alive_snakes = [id for id in self._snakes.keys() if id not in self._dead_snakes]
        ids_to_batch = list(alive_snakes)
        batches: List[List[int]] = []
        while ids_to_batch:
            current_id = ids_to_batch.pop()
            for batch in batches:
                if can_go_in_batch(batch, current_id):
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
