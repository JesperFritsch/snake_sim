import time
import json
import logging
from importlib import resources
from pathlib import Path
from typing import List, Dict
from multiprocessing import Event

from snake_sim.utils import profile
from snake_sim.environment.snake_env import EnvStepData
from snake_sim.environment.interfaces.main_loop_interface import IMainLoop
from snake_sim.environment.interfaces.loop_observer_interface import ILoopObserver
from snake_sim.environment.snake_handlers import ISnakeHandler
from snake_sim.environment.interfaces.snake_env_interface import ISnakeEnv
from snake_sim.environment.types import (
    LoopStartData,
    LoopStepData,
    LoopDecisionData,
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
        self._decision_timeout_s = None
        self._steps = 0
        self._current_step_data: LoopStepData = None
        self._step_start_time = None
        self._is_running = False
        self._did_notify_start = False
        self._did_notify_stop = False
        self._end_on_last_standing_when_longest = False
        self._end_on_last_standing_buffer_steps: int | None = None
        self._alone_and_longest_since: int | None = None
        self._end_when_dead_tag: str | None = None
        self._end_when_dead_buffer_steps = 0
        self._watched_snake_id: int | None = None
        self._watched_died_at_step: int | None = None

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

    def _prepare_batch(self, batch: List[int]) -> Dict[int, EnvStepData]:
        return {id: self._env.get_env_step_data(id) for id in batch}

    def _update_batch(self, batch: List[int]):
        batch_data = self._prepare_batch(batch)
        decisions = self._snake_handler.get_decisions(
            batch_data,
            self._decision_timeout_s,
            on_response=self._decision_callback,
        )
        self._apply_decisions(decisions)

    def _apply_decisions(self, decisions: Dict[int, Coord]):
        for id, decision in decisions.items():
            if decision is None:
                decision = Coord(0, 0) # No move
            alive, grew, tail_direction = self._env.move_snake(id, decision)
            if not alive:
                self._snake_handler.kill_snake(id)
                decision = Coord(0, 0)  # No move if dead
            self._current_step_data.alive_states[id] = alive
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
            self._notify_start(LoopStartData(env_meta_data=self._env.get_init_data()))
            self._loop()
        finally:
            self._notify_stop(LoopStopData(final_step=self._steps))

    def stop(self):
        self._is_running = False
        super().stop()

    def _pre_update(self):
        self._env.update_food()
        self._current_step_data = LoopStepData(
            step=0,
            total_time=0,
            alive_states={},
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
        snakes = self._env.get_env_step_data().snakes
        self._current_step_data.lengths = {id: snake['length'] for id, snake in snakes.items()}
        self._current_step_data.total_time = total_time
        for sid in self._current_step_data.decisions:
            self._current_step_data.snake_times.setdefault(sid, total_time)

        if self._end_on_last_standing_when_longest:
            alive_ids = [sid for sid, s in snakes.items() if s['is_alive']]
            if len(alive_ids) == 1:
                longest = max(s['length'] for s in snakes.values())
                if snakes[alive_ids[0]]['length'] >= longest:
                    if self._alone_and_longest_since is None:
                        self._alone_and_longest_since = self._steps
                    buffer = self._end_on_last_standing_buffer_steps or 0
                    if self._steps - self._alone_and_longest_since >= buffer:
                        self.stop()

        if self._end_when_dead_tag is not None:
            if self._watched_snake_id is None and self._watched_died_at_step is None:
                for sid in self._snake_handler.get_snakes():
                    if self._snake_handler.get_snake_tag(sid) == self._end_when_dead_tag:
                        self._watched_snake_id = sid
                        break
                else:
                    log.warning(
                        f"end-when-dead tag {self._end_when_dead_tag!r} matches no snake — disabling"
                    )
                    self._end_when_dead_tag = None
            if (
                self._watched_snake_id is not None
                and self._watched_died_at_step is None
                and not snakes.get(self._watched_snake_id, {}).get('is_alive', True)
            ):
                self._watched_died_at_step = self._steps
            if (
                self._watched_died_at_step is not None
                and self._steps - self._watched_died_at_step >= self._end_when_dead_buffer_steps
            ):
                self.stop()

        if len(self._current_step_data.decisions):
            self._notify_step(self._current_step_data)
    
    def _decision_callback(self, snake_id: int, wall_time_ns: int):
        wall_time_s = wall_time_ns / 1e9
        self._current_step_data.snake_times[snake_id] = wall_time_s
        return super()._notify_decision(
            LoopDecisionData(
                snake_id=snake_id, 
                step_idx=self._current_step_data.step, 
                wall_time_ns=wall_time_ns
            )
        )

    def set_snake_handler(self, snake_handler: ISnakeHandler):
        self._snake_handler = snake_handler

    def set_max_no_food_steps(self, steps):
        self._max_no_food_steps = steps

    def set_max_steps(self, steps):
        self._max_steps = steps

    def set_end_on_last_standing_when_longest(self, enabled: bool):
        self._end_on_last_standing_when_longest = enabled

    def set_end_on_last_standing_buffer_steps(self, buffer_steps: int | None):
        self._end_on_last_standing_buffer_steps = (
            max(0, buffer_steps) if buffer_steps is not None else None
        )
        self._alone_and_longest_since = None

    def set_end_when_dead_tag(self, tag: str | None, buffer_steps: int = 0):
        self._end_when_dead_tag = tag
        self._end_when_dead_buffer_steps = max(0, buffer_steps)
        self._watched_snake_id = None
        self._watched_died_at_step = None

    def set_environment(self, env: ISnakeEnv):
        self._env = env

    def set_decision_timout(self, timeout_ms: int):
        if timeout_ms is not None:
            self._decision_timeout_s = timeout_ms / 1000


class GameLoop(SimLoop):

    def __init__(self):
        super().__init__()
        self._steps_per_sec = None
        self._last_update_time = None

    def _post_update(self):
        # Using the post update hook to control the speed of the game
        super()._post_update()
        sleep_time = 1 / self._steps_per_sec
        if not self._last_update_time is None:
            sleep_time -= time.time() - self._last_update_time
        if sleep_time > 0:
            time.sleep(sleep_time)

    def start(self):
        if not self._steps_per_sec:
            raise ValueError('Steps per minute not set')
        super().start()

    def set_steps_per_sec(self, spm):
        self._steps_per_sec = spm

