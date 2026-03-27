import argparse
import sys
import json
import numpy as np
import math

from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict

import snake_sim.debugging as debug

from snake_sim.loop_observables.file_reader_observable import FileRepeaterObservable
from snake_sim.loop_observers.state_builder_observer import StateBuilderObserver
from snake_sim.loop_observers.map_builder_observer import MapBuilderObserver
from snake_sim.loop_observers.waitable_observer import WaitableObserver
from snake_sim.map_utils.general import print_map
from snake_sim.environment.types import (
    Coord, 
    CompleteStepState, 
    EnvMetaData,
    NoMoreSteps,
    AreaCheckResult
)

from snake_sim.cpp_bindings.utils import (
    distance_to_coord, 
    distance_to_tile_with_value,
    get_visitable_tiles,
    area_boundary_tiles
)

from snake_sim.cpp_bindings.area_check import AreaChecker


AREA_CHECKERS = {}


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
class TrapInfo:
    trapped_ids: set[int] = field(default_factory=set)
    trapping_ids: set[int] = field(default_factory=set)

    def __hash__(self):
        return hash((frozenset(self.trapped_ids), frozenset(self.trapping_ids)))

    def to_dict(self):
        return {
            "trapped_ids": list(self.trapped_ids),
            "trapping_ids": list(self.trapping_ids)
        }


@dataclass
class RunAnalysis:
    run_file: Path
    env_meta_data: EnvMetaData
    snake_ids: list[int]
    fatal_steps: dict[int, int] # snake_id -> step_idx of fatal step
    final_step_idx: int
    tail_visible_phases: dict[int, list[TailVisiblePhase]] # snake_id -> list of phases where it can see its tail or not
    entered_separate_area: dict[int, list[int]] # snake_id -> list of step idx where it entered a separate area (cannot reach its body anymore)
    traps_mapping: dict[int, set[TrapInfo]] = field(default_factory=dict) # step_idx -> set of TrapInfo for traps detected in that step

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
            "entered_separate_area": self.entered_separate_area,
            "traps_mapping": {
                step_idx: [trap_info.to_dict() for trap_info in trap_infos]
                for step_idx, trap_infos in self.traps_mapping.items()
            }
        }


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


def get_area_checkers(
        snake_values: dict[int, dict[str, int]], 
        free_value: int, 
        food_value: int, 
        width: int,
        height: int,
    ) ->  dict[int, AreaChecker]:
    global AREA_CHECKERS
    if not AREA_CHECKERS:
        debug.debug_print("Initializing area checkers...")
    else:
        return AREA_CHECKERS
    for snake_id, values in snake_values.items():
        if snake_id not in AREA_CHECKERS:
            AREA_CHECKERS[snake_id] = AreaChecker(
                food_value,
                free_value,
                values['body_value'],
                values['head_value'],
                width,
                height,
            )
    return AREA_CHECKERS


def get_best_area_checks(
        area_checkers: dict[int, AreaChecker],
        state: CompleteStepState, 
        s_map: np.ndarray,
        ids_to_check: set[int]
    ) -> dict[int, AreaCheckResult]:
    """ Check if each snake can access the area around it. """
    best_area_checks: dict[int, AreaCheckResult] = {}
    for snake_id, body in state.snake_bodies.items():
        if snake_id not in ids_to_check:
            continue
        head_coord = body[0]
        visitable_tiles = get_visitable_tiles(
            s_map,
            state.env_meta_data.width,
            state.env_meta_data.height,
            head_coord,
            [state.env_meta_data.free_value, state.env_meta_data.food_value]
        )
        target_margin = max(10, math.ceil(0.06 * len(body)))
        results = [
            area_checkers[snake_id].area_check(
                s_map,
                list(body),
                tile,
                target_margin=target_margin,
                food_check=False,
                complete_area=False,
                exhaustive=False
            ) for tile in visitable_tiles
        ]

        best_area_checks[snake_id] = max(
            (AreaCheckResult(**result) for result in results),
            key=lambda r: r.margin_frac
        ) if results else AreaCheckResult(False, 0, 0, 0, False, -1, 0.0)

    return best_area_checks


def find_trapped_candidates(
    area_checks1: dict[int, AreaCheckResult],
    area_checks2: dict[int, AreaCheckResult],
    trap_threshold: float = 0.8
) -> set[int]:
    """ Detect snakes that have recently become trapped.
    A snake is considered recently trapped if its area check margin has dropped by more then the specified threshold percentage.
    
    args:
        area_checks1: Area check results for the previous step.
        area_checks2: Area check results for the current step.
        trap_threshold: The threshold for considering a snake as recently trapped based on margin drop percentage.

    returns:         A set of snake IDs that are considered trapped between the two checks.

    """
    trapped_snakes = set()
    for snake_id, check2 in area_checks2.items():
        check1 = area_checks1.get(snake_id)
        if check1 is not None and check2 is not None:
            if ((check2.margin_frac < check1.margin_frac * (1 - trap_threshold)) # if margin dropped significantly check if some snake contributed to the trapping
                or (check1.margin >= 0 and check2.margin < 0)):
                trapped_snakes.add(snake_id)
    return trapped_snakes


def create_traps_set(
    trapped_snakes: set[int],
    state: CompleteStepState,
    s_map: np.ndarray,
    snake_ids: set[int]
) -> set[TrapInfo]:

    if state is None or not trapped_snakes:
        return {}
    
    trapping_area_boundaries: dict[int, set[int]] = {}


    for snake_id in trapped_snakes:
        head_coord = state.snake_bodies[snake_id][0]
        boundary_tiles = area_boundary_tiles(
            s_map,
            state.env_meta_data.width,
            state.env_meta_data.height,
            state.env_meta_data.free_value,
            head_coord
        )
        trapping_area_boundaries[snake_id] = set(boundary_tiles)
    traps: set[TrapInfo] = set()
    for trapping_id in snake_ids:
        if not state.snake_alive.get(trapping_id, False):
            continue
        head_value = state.env_meta_data.snake_values[trapping_id]['head_value']
        for trapped_id, boundary_tiles in trapping_area_boundaries.items():
            debug.debug_print("boarding_values: ", head_value, boundary_tiles)
            if trapping_id == trapped_id:
                continue
            trapped_head_coord = state.snake_bodies[trapped_id][0]
            trapping_visitable_tiles = get_visitable_tiles(
                s_map,
                state.env_meta_data.width,
                state.env_meta_data.height,
                state.snake_bodies[trapping_id][0],
                [state.env_meta_data.free_value, state.env_meta_data.food_value]
            )
            distances_to_trapped_head = [distance_to_coord(
                s_map,
                state.env_meta_data.width,
                state.env_meta_data.height,
                coord,
                tuple(trapped_head_coord),
                [state.env_meta_data.free_value, state.env_meta_data.food_value]
            ) for coord in trapping_visitable_tiles]
            debug.debug_print(f"Distances from snake {trapping_id} to trapped snake {trapped_id} head: {distances_to_trapped_head}")
            if (head_value in boundary_tiles and 
                any(d < 0 for d in distances_to_trapped_head) and 
                trapping_id not in trapped_snakes):
                # If the trapping snake's head is on the boundary of the trapped area 
                # and at least one of the visitable tiles from the trapping snake can not reach the trapped snake's head,
                # and the trapping snake is not itself trapped, we consider it a contributor to the trapping. 
                traps.add(TrapInfo(trapped_ids={trapped_id}, trapping_ids={trapping_id}))

    # group by trapping and trapped snakes, so we can see which snakes are commonly trapping the same snakes
    #first group by trapped snake, then by trapping snakes

    partially_grouped: set[TrapInfo] = set()
    accumulate: defaultdict[set[int], TrapInfo] = defaultdict(TrapInfo)
    for trap_info in traps:
        for trapped_id in trap_info.trapped_ids:
            accumulate[trapped_id].trapped_ids = {trapped_id}
            accumulate[trapped_id].trapping_ids.update(trap_info.trapping_ids)
    partially_grouped = set(accumulate.values())
    grouped_trapped_mapping: set[TrapInfo] = set()
    accumulate: defaultdict[set[int], TrapInfo] = defaultdict(TrapInfo)
    for trap_info in partially_grouped:
        accumulate[tuple(trap_info.trapping_ids)].trapping_ids = trap_info.trapping_ids
        accumulate[tuple(trap_info.trapping_ids)].trapped_ids.update(trap_info.trapped_ids)
    grouped_trapped_mapping = set(accumulate.values())
                
    return grouped_trapped_mapping


def find_traps(
        prev_state: CompleteStepState, 
        current_state: CompleteStepState, 
        prev_map: np.ndarray, 
        current_map: np.ndarray, 
        snake_ids: list[int]
    ) -> dict[int, set[TrapInfo]]:
    get_area_checkers(
        snake_values=prev_state.env_meta_data.snake_values,
        free_value=prev_state.env_meta_data.free_value,
        food_value=prev_state.env_meta_data.food_value,
        width=prev_state.env_meta_data.width,
        height=prev_state.env_meta_data.height,
    )
    ids_to_check_for_traps = set([id for id, alive in prev_state.snake_alive.items() if alive])
    best_area_checks_s1 = get_best_area_checks(AREA_CHECKERS, prev_state, prev_map, ids_to_check_for_traps)
    best_area_checks_s2 = get_best_area_checks(AREA_CHECKERS, current_state, current_map, ids_to_check_for_traps)

    trapped_snakes = find_trapped_candidates(best_area_checks_s1, best_area_checks_s2)
    return create_traps_set(trapped_snakes, current_state, current_map, snake_ids)


def cli(argv):
    ap = argparse.ArgumentParser(description="Run the analyzer on a run file")
    ap.add_argument("filepath", type=Path, help="Path to the run file to analyze")
    ap.add_argument("--output", "-o", type=Path, help="Path to save the analysis result as json", default=Path("run_analysis.json"))
    return ap.parse_args(argv)


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
    traps_mapping: dict[int, set[TrapInfo]] = defaultdict(set) # step_idx -> set of TrapInfo for traps detected in that step
    prev_state = state_builder.get_state(0)
    prev_map = map_builder.get_map(0)
    while True:
        try:
            current_state = state_builder.get_next_state()
            current_step_idx = state_builder.get_current_step_idx()
            current_map = map_builder.get_map_for_step(current_step_idx)

            traps_found = find_traps(prev_state, current_state, prev_map, current_map, snake_ids) if len(snake_ids) > 1 else {}
            if traps_found:
                traps_mapping[current_step_idx].update(traps_found)
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
            prev_state = current_state
            prev_map = current_map
        except StopIteration:
            break
        except NoMoreSteps:
            continue
            
    # clean up entered area lists, remove values incrementing by 1 from the end
    for s_id, step_idxs in entered_separate_area_dict.items():
        cleaned_step_idxs = []
        for idx in step_idxs:
            if len(cleaned_step_idxs) > 0 and idx == cleaned_step_idxs[-1] + 1:
                continue
            cleaned_step_idxs.append(idx)
        entered_separate_area_dict[s_id] = cleaned_step_idxs
        
    analysis = RunAnalysis(
        run_file=args.filepath,
        env_meta_data=env_meta_data,
        snake_ids=snake_ids,
        fatal_steps=fatal_steps,
        tail_visible_phases=tail_visible_phases,
        final_step_idx=current_step_idx,
        entered_separate_area=entered_separate_area_dict,
        traps_mapping=traps_mapping
    )
    with open(args.output, "w") as f:
        json.dump(analysis.to_dict(), f, indent=2)


if __name__ == "__main__":
    args = cli(sys.argv[1:])
    debug.enable_debug_for_all()
    # debug.activate_debug()
    main(args)