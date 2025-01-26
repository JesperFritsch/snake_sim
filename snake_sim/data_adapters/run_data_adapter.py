import json

from importlib import resources
from typing import Dict

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
        self._run_data = None
        self.snake_reps: Dict[int, SnakeRep] = {}
        self.initialize(env_init_data, color_mapping)

    def initialize(self, env_init_data: EnvInitData, color_mapping: dict):
        self._run_data = RunData(
            height=env_init_data.height,
            width=env_init_data.width,
            color_mapping=color_mapping,
            base_map=env_init_data.base_map,
            food_value=env_init_data.food_value,
            free_value=env_init_data.free_value,
            blocked_value=env_init_data.blocked_value,
            snake_ids=list(env_init_data.snake_values.keys()),
            snake_values=env_init_data.snake_values
        )
        self.snake_reps: Dict[int, SnakeRep] = {}
        for id, snake_data in env_init_data.snake_values.items():
            start_pos = env_init_data.start_positions[id]
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
            # Grow has to be called before move
            if loop_step.snake_grew.get(snake_id, False):
                snake_rep.grow()
            self.snake_reps[snake_id].move(decision)
            step_data.add_snake_data(snake_id, snake_rep.body)
            self._run_data.add_step(step_data)
        return step_data

