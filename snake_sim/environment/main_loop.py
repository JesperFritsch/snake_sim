import time
import json
from dataclasses import dataclass, field
from importlib import resources

from typing import Optional, List, Dict
from multiprocessing import Event

from snake_sim.environment.interfaces.main_loop_interface import IMainLoop
from snake_sim.environment.interfaces.loop_observer_interface import ILoopObserver
from snake_sim.utils import DotDict, Coord, profile
from snake_sim.environment.snake_handlers import ISnakeHandler
from snake_sim.environment.interfaces.snake_env_interface import ISnakeEnv

with resources.open_text('snake_sim.config', 'default_config.json') as config_file:
    config = DotDict(json.load(config_file))


@dataclass
class LoopStepData:
    step: int
    total_time: Optional[float] = field(default_factory=float)
    snake_times: Optional[Dict[int, float]] = field(default_factory=dict)
    decisions: Optional[Dict[int, Coord]] = field(default_factory=dict)
    snake_grew: Optional[Dict[int, bool]] = field(default_factory=dict)
    lengths: Optional[Dict[int, int]] = field(default_factory=dict)
    food: Optional[List[Coord]] = field(default_factory=list)


class SimLoop(IMainLoop):

    def __init__(self):
        self._snake_handler: ISnakeHandler = None
        self._observers: List[ILoopObserver] = []
        self._env: ISnakeEnv = None
        self._max_no_food_steps = None
        self._max_steps = None
        self._steps = 0
        self._current_step_data: LoopStepData = None
        self._step_start_time = None
        self._is_running = False
        self._did_notify_start = False
        self._did_notify_end = False

    # @profile()
    def _loop(self):
        while self._is_running:
            snake_positions = self._env.get_head_positions()
            update_batches = self._snake_handler.get_batch_order(snake_positions)
            if len(update_batches) == 0:
                self.stop()
            self._pre_update()
            self._env.update_food()
            self._current_step_data.food = self._env.get_food()
            for batch in update_batches:
                decisions = self._snake_handler.get_decisions({id: self._env.get_env_data(id) for id in batch})
                for id, decision in decisions.items():
                    if decision is None:
                        self._snake_handler.kill_snake(id)
                    else:
                        alive, grew = self._env.move_snake(id, decision)
                        if alive:
                            self._current_step_data.snake_times[id] = 0 # TODO: Implement snake times
                            self._current_step_data.decisions[id] = decision
                            self._current_step_data.snake_grew[id] = grew
                        else:
                            self._snake_handler.kill_snake(id)
            self._steps += 1
            self._post_update()

    def start(self):
        self._did_notify_start = False
        self._did_notify_end = False
        self._is_running = True
        if not self._snake_handler:
            raise ValueError('Snake handler not set')
        if not self._env:
            raise ValueError('Environment not set')
        try:
            self._notify_start()
            self._loop()
        finally:
            self._notify_end()

    def stop(self):
        self._notify_end()
        self._is_running = False

    def _pre_update(self):
        self._current_step_data = LoopStepData(self._steps)
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

    def _notify_start(self):
        if self._did_notify_start: return
        for observer in self._observers:
            observer.notify_start()
        self._did_notify_start = True

    def _notify_step(self):
        for observer in self._observers:
            observer.notify_step(self._current_step_data)

    def _notify_end(self):
        if self._did_notify_end: return
        for observer in self._observers:
            observer.notify_end()
        self._did_notify_end = True

    def set_snake_handler(self, snake_handler: ISnakeHandler):
        self._snake_handler = snake_handler

    def set_max_no_food_steps(self, steps):
        self._max_no_food_steps = steps

    def set_max_steps(self, steps):
        self._max_steps = steps

    def add_observer(self, observer: ILoopObserver):
        self._observers.append(observer)

    def get_observers(self) -> List[ILoopObserver]:
        return self._observers

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

