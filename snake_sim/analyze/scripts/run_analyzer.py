import argparse
import sys
import json
import numpy as np

from pathlib import Path

from dataclasses import dataclass
from snake_sim.loop_observables.file_reader_observable import FileRepeaterObservable
from snake_sim.loop_observers.state_builder_observer import StateBuilderObserver
from snake_sim.loop_observers.map_builder_observer import MapBuilderObserver
from snake_sim.loop_observers.waitable_observer import WaitableObserver
from snake_sim.map_utils.general import print_map
from snake_sim.environment.types import (
    Coord, 
    CompleteStepState, 
    EnvMetaData,
    NoMoreSteps
)

from snake_sim.cpp_bindings.utils import (
    distance_to_coord, 
    distance_to_tile_with_value,
    get_visitable_tiles,
)


@dataclass
class TailVisiblePhase:
    start_step_idx: int
    end_step_idx: int | None = None
    tail_visible: bool = True

    def to_dict(self):
        return {
            "start_step_idx": self.start_step_idx,
            "end_step_idx": self.end_step_idx,
            "tail_visible": self.tail_visible
        }
    

@dataclass
class RunAnalisis:
    run_file: Path
    env_meta_data: EnvMetaData
    snake_ids: list[int]
    fatal_steps: dict[int, int] # snake_id -> step_idx of fatal step
    final_step_idx: int
    tail_visible_phases: dict[int, list[TailVisiblePhase]] # snake_id -> list of phases where it can see its tail or not
    entered_separate_area: dict[int, list[int]] # snake_id -> list of step idx where it entered a separate area (cannot reach its body anymore)

    def to_dict(self):
        return {
            "run_file": str(self.run_file),
            "env_meta_data": self.env_meta_data.to_dict(),
            "snake_ids": self.snake_ids,
            "fatal_steps": self.fatal_steps,
            "final_step_idx": self.final_step_idx,
            "tail_visible_phases": {
                s_id: [phase.to_dict() for phase in phases] 
                for s_id, phases in self.tail_visible_phases.items()
            },
            "entered_separate_area": self.entered_separate_area
        }


def cli(argv):
    ap = argparse.ArgumentParser(description="Run the analyzer on a run file")
    ap.add_argument("filepath", type=Path, help="Path to the run file to analyze")
    return ap.parse_args(argv)


def can_see_tail(state: CompleteStepState, s_map: np.ndarray, snake_id: int) -> bool:
    snake_body = state.snake_bodies[snake_id]
    visitable_values = [state.env_meta_data.free_value, state.env_meta_data.food_value]
    height, width = s_map.shape
    head = snake_body[0]
    tail = snake_body[-1]
    dist_to_tail = distance_to_coord(
        s_map,
        width,
        height, 
        tuple(head), 
        tuple(tail),
        visitable_values
    )
    return dist_to_tail >= 0

def entered_separate_area(state: CompleteStepState, s_map: np.ndarray, snake_id: int) -> bool:
    snake_body = state.snake_bodies[snake_id]
    visitable_values = [state.env_meta_data.free_value, state.env_meta_data.food_value]
    height, width = s_map.shape
    head = snake_body[0]
    body_value = state.env_meta_data.snake_values[snake_id]['body_value']
    visitable_tiles = get_visitable_tiles(s_map, width, height, Coord(*head), visitable_values)
    dists_to_body = [distance_to_tile_with_value(
        s_map,
        width,
        height,
        visitable,
        body_value,
        visitable_values
    ) for visitable in visitable_tiles]
    return visitable_tiles and all(dist == -1  for dist in dists_to_body)

def main(args):
    loop_repeater = FileRepeaterObservable(filepath=args.filepath)

    state_builder = StateBuilderObserver()
    map_builder = MapBuilderObserver()
    waitable_observer = WaitableObserver()

    loop_repeater.add_observer(state_builder)
    loop_repeater.add_observer(map_builder)
    loop_repeater.add_observer(waitable_observer)

    loop_repeater.start()

    waitable_observer.wait_until_started()

    start_data = state_builder.get_start_data()
    snake_ids = list(start_data.env_meta_data.snake_values.keys())
    env_meta_data = start_data.env_meta_data
    fatal_steps = {}
    tail_visible_phases: dict[int, list[TailVisiblePhase]] = {s_id: [] for s_id in snake_ids}
    entered_separate_area_dict: dict[int, list[int]] = {s_id: [] for s_id in snake_ids}
    while True:
        try:
            current_state = state_builder.get_next_state()
            current_step_idx = state_builder.get_current_step_idx()
            current_map = map_builder.get_map_for_step(current_step_idx)
            for s_id in snake_ids:
                tail_visible = can_see_tail(current_state, current_map, s_id)
                if len(tail_visible_phases[s_id]) == 0:
                    tail_visible_phases[s_id].append(TailVisiblePhase(start_step_idx=current_step_idx, tail_visible=tail_visible))
                else:
                    last_phase = tail_visible_phases[s_id][-1]
                    if tail_visible != last_phase.tail_visible:
                        last_phase.end_step_idx = current_step_idx - 1
                        tail_visible_phases[s_id].append(TailVisiblePhase(start_step_idx=current_step_idx, tail_visible=tail_visible))

                if not current_state.snake_alive[s_id] and s_id not in fatal_steps:
                    fatal_steps[s_id] = current_step_idx
                if current_state.snake_alive[s_id] and entered_separate_area(current_state, current_map, s_id):
                    entered_separate_area_dict[s_id].append(current_step_idx)
        except StopIteration:
            break
        except NoMoreSteps:
            continue
    # clean up entered area lists, remove values incrementing by 1 from the end
    for s_id, step_idxs in entered_separate_area_dict.items():
        cleaned_step_idxs = []
        first_in_seq = None
        for idx in step_idxs:
            if len(cleaned_step_idxs) > 0 and idx == cleaned_step_idxs[-1] + 1:
                continue
            cleaned_step_idxs.append(idx)
            first_in_seq = idx
        entered_separate_area_dict[s_id] = cleaned_step_idxs
    analysis = RunAnalisis(
        run_file=args.filepath,
        env_meta_data=env_meta_data,
        snake_ids=snake_ids,
        fatal_steps=fatal_steps,
        tail_visible_phases=tail_visible_phases,
        final_step_idx=current_step_idx,
        entered_separate_area=entered_separate_area_dict
    )
    print(json.dumps(analysis.to_dict(), indent=2))


if __name__ == "__main__":
    args = cli(sys.argv[1:])
    main(args)