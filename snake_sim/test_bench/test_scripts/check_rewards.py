from pathlib import Path
from typing import Dict

from snake_sim.loop_observables.file_reader_observable import FileRepeaterObservable
from snake_sim.loop_observers.state_builder_observer import StateBuilderObserver, NoMoreSteps
from snake_sim.loop_observers.map_builder_observer import MapBuilderObserver
from snake_sim.loop_observers.waitable_observer import WaitableObserver
from snake_sim.environment.snake_env import SnakeRep
from collections import deque
from snake_sim.environment.types import Coord, CompleteStepState
from snake_sim.render.utils import create_color_map
from snake_sim.rl.reward import compute_rewards

FILE_PATH = Path("/home/jesper/dev/snake_sim/snake_sim/runs/grid_32x32/run_32x32_10snakes_2253steps.run_proto")

def rgb_color_text(text, r, g, b):
    return f"\033[48;2;{r};{g};{b}m{text}\033[0m"

def create_snake_reps(step_state: CompleteStepState) -> Dict[int, SnakeRep]:
    snake_reps = {}
    for snake_id, snake_body in step_state.snake_bodies.items():
        snake_id_int = int(snake_id)
        body_val = step_state.env_meta_data.snake_values[snake_id]["body_value"]
        head_val = step_state.env_meta_data.snake_values[snake_id]["head_value"]
        snake_rep = SnakeRep(snake_id_int, head_val, body_val, Coord(0, 0))
        snake_rep._length = len(snake_body)
        snake_rep.body = deque([Coord(*coord) for coord in snake_body])
        snake_reps[snake_rep.id] = snake_rep
    return snake_reps

loop_repeater = FileRepeaterObservable(filepath=FILE_PATH)

state_builder = StateBuilderObserver()
map_builder = MapBuilderObserver()
waitable_observer = WaitableObserver()

loop_repeater.add_observer(state_builder)
loop_repeater.add_observer(map_builder)
loop_repeater.add_observer(waitable_observer)

loop_repeater.start()

waitable_observer.wait_until_started()

prev_state = state_builder.get_state(0)
prev_map = map_builder.get_map(0)
start_data = state_builder.get_start_data()
snake_ids = start_data.env_meta_data.snake_values.keys()

rewards_per_step = {}



snake_reps = create_snake_reps(prev_state)
color_mapping = create_color_map(prev_state.env_meta_data.snake_values)
for s_rep in snake_reps.values():
    print(f"ID: {s_rep.id: <4} HEAD: {Coord(*s_rep.get_head()): <20} body len: {s_rep._length: <4}, body_color: {rgb_color_text('  ', *color_mapping[s_rep.body_value])}")

while True:
    try:
        while True:
            current_state = state_builder.get_next_state()
            current_map = map_builder.get_next_map()
            current_step_idx = state_builder.get_current_step_idx()
            rewards, info = compute_rewards(
                (prev_state, prev_map),
                (current_state, current_map),
                snake_ids
            )
            rewards_per_step[current_step_idx] = info
            prev_state = current_state
    except StopIteration:
        break
    except NoMoreSteps:
        continue


print(len(rewards_per_step))
print(len(rewards_per_step[1]))
print([k for k, v in rewards_per_step.items() if any(r["trapping_reward"] > 0 for r in v.values())])
print({k: list(map(lambda x: x[0], filter(lambda r: r[1]["trapping_reward"] > 0, v.items()))) for k, v in rewards_per_step.items() if any(r["trapping_reward"] > 0 for r in v.values())})
