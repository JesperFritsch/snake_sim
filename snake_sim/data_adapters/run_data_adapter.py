import json

from importlib import resources

from snake_sim.environment.snake_env import EnvInitData, SnakeRep
from snake_sim.environment.main_loop import LoopStepData
from snake_sim.run_data.run_data import RunData, StepData
from snake_sim.utils import DotDict, Coord

with resources.open_text('snake_sim.config', 'default_config.json') as config_file:
    config = DotDict(json.load(config_file))

class RunDataAdapter:
    def __init__(self, env_init_data: EnvInitData, color_mapping: dict):
        if not isinstance(env_init_data, EnvInitData):
            raise ValueError('env_init_data must be of type EnvInitData')
        self._env_init_data = env_init_data
        self._run_data = RunData(
            height=self._env_init_data.height,
            width=self._env_init_data.width,
            color_mapping=color_mapping,
            base_map=self._env_init_data.base_map,
            food_value=self._env_init_data.food_value,
            free_value=self._env_init_data.free_value,
            blocked_value=self._env_init_data.blocked_value,
            snakes=list(self._env_init_data.snake_values.keys()),
            snake_values=self._env_init_data.snake_values
        )
        self.snake_reps = {}
        for id, snake_data in self._env_init_data.snake_values.items():
            start_pos = self._env_init_data.start_positions[id]
            self.snake_reps[id] = SnakeRep(id, snake_data['head_value'], snake_data['body_value'], start_pos)

    def get_run_data(self):
        return self._run_data

    def get_metadata(self):
        return self._run_data.get_metadata()

    def loop_step_data_to_step_data(self, loop_step: LoopStepData) -> StepData:
        if not isinstance(loop_step, LoopStepData):
            raise ValueError('loop_step must be of type LoopStepData')
        step_data = StepData(
            step=loop_step.step,
            food=loop_step.food
        )
        for snake_id, decision in loop_step.desicions.items():
            snake_rep = self.snake_reps[snake_id]
            grow = len(snake_rep.body) < loop_step.lengths[snake_id]
            self.snake_reps[snake_id].move(decision, grow=grow)
            step_data.add_snake_data(snake_id, snake_rep.body, did_grow=grow)
            self._run_data.add_step(step_data)
        return step_data

