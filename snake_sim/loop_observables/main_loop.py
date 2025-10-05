import time
import json
import logging
from importlib import resources
from pathlib import Path
from typing import List, Dict
from multiprocessing import Event

from snake_sim.utils import profile
from snake_sim.environment.snake_env import EnvData
from snake_sim.environment.interfaces.main_loop_interface import IMainLoop
from snake_sim.environment.interfaces.loop_observer_interface import ILoopObserver
from snake_sim.environment.snake_handlers import ISnakeHandler
from snake_sim.environment.interfaces.snake_env_interface import ISnakeEnv
from snake_sim.environment.types import (
    LoopStartData,
    LoopStepData,
    LoopStopData,
    DotDict,
    Coord
)

with resources.open_text('snake_sim.config', 'default_config.json') as config_file:
    config = DotDict(json.load(config_file))

log = logging.getLogger(Path(__file__).stem)


class SimLoop(IMainLoop):

    def __init__(self):
        super().__init__()
        self._snake_handler: ISnakeHandler = None
        self._env: ISnakeEnv = None
        self._max_no_food_steps = None
        self._max_steps = None
        self._steps = 0
        self._current_step_data: LoopStepData = None
        self._step_start_time = None
        self._is_running = False
        self._did_notify_start = False
        self._did_notify_stop = False

    # @profile("cumtime")
    def _loop(self):
        while self._is_running:
            snake_positions = self._env.get_head_positions()
            update_batches = self._snake_handler.get_batch_order(snake_positions)
            if len(update_batches) == 0:
                self.stop()
            self._pre_update()
            for batch in update_batches:
                self._update_batch(batch)
            self._steps += 1
            self._post_update()

    def _prepare_batch(self, batch: List[int]) -> Dict[int, EnvData]:
        return {id: self._env.get_env_data(id) for id in batch}

    def _update_batch(self, batch: List[int]):
        batch_data = self._prepare_batch(batch)
        decisions = self._snake_handler.get_decisions(batch_data)
        self._apply_decisions(decisions)

    def _apply_decisions(self, decisions: Dict[int, Coord]):
        for id, decision in decisions.items():
            if decision is None:
                decision = Coord(0, 0) # No move
            alive, grew, tail_direction = self._env.move_snake(id, decision)
            if not alive:
                self._snake_handler.kill_snake(id)
            self._current_step_data.snake_times[id] = 0 # TODO: Implement snake times
            self._current_step_data.decisions[id] = decision
            self._current_step_data.tail_directions[id] = tail_direction
            self._current_step_data.snake_grew[id] = grew

    def start(self):
        self._is_running = True
        if not self._snake_handler:
            raise ValueError('Snake handler not set')
        if not self._env:
            raise ValueError('Environment not set')
        try:
            self._notify_start()
            self._loop()
        finally:
            self._notify_stop()

    def stop(self):
        self._is_running = False

    def _pre_update(self):
        self._env.update_food()
        self._current_step_data = DotDict(
            step=0,
            total_time=0,
            snake_times={},
            decisions={},
            tail_directions={},
            snake_grew={},
            lengths={},
            new_food=[],
            removed_food=[]
        )
        self._current_step_data.step = self._steps
        new_food, removed_food = self._env.get_food_diff()
        self._current_step_data.new_food = new_food
        self._current_step_data.removed_food = removed_food
        self._step_start_time = time.time()

    def _post_update(self):
        if self._max_no_food_steps and self._env.steps_since_any_ate() > self._max_no_food_steps:
            self.stop()
        if self._max_steps is not None and self._steps > self._max_steps:
            self.stop()
        total_time = time.time() - self._step_start_time
        self._current_step_data.lengths = {id: snake['length'] for id, snake in self._env.get_env_data().snakes.items()}
        self._current_step_data.total_time = total_time
        self._notify_step()

    def _get_start_data(self) -> LoopStartData:
        return LoopStartData(
            env_init_data=self._env.get_init_data()
        )

    def _get_step_data(self) -> LoopStepData:
        return LoopStepData(**self._current_step_data)

    def _get_stop_data(self) -> LoopStopData:
        return LoopStopData(final_step=self._steps)

    def set_snake_handler(self, snake_handler: ISnakeHandler):
        self._snake_handler = snake_handler

    def set_max_no_food_steps(self, steps):
        self._max_no_food_steps = steps

    def set_max_steps(self, steps):
        self._max_steps = steps

    def set_environment(self, env: ISnakeEnv):
        self._env = env


class GameLoop(SimLoop):

    def __init__(self):
        super().__init__()
        self._steps_per_min = None
        self._last_update_time = None

    def _post_update(self):
        # Using the post update hook to control the speed of the game
        super()._post_update()
        sleep_time = 60 / self._steps_per_min
        if not self._last_update_time is None:
            sleep_time -= time.time() - self._last_update_time
        if sleep_time > 0:
            time.sleep(sleep_time)

    def start(self):
        if not self._steps_per_min:
            raise ValueError('Steps per minute not set')
        super().start()

    def set_steps_per_min(self, spm):
        self._steps_per_min = spm
