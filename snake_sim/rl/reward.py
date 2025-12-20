
import math
import numpy as np
from typing import Dict

import snake_sim.debugging as debug
from snake_sim.map_utils.general import print_map
from snake_sim.environment.types import CompleteStepState
from snake_sim.environment.types import AreaCheckResult
from snake_sim.rl.state_builder import print_state
from snake_sim.cpp_bindings.utils import distance_to_tile_with_value
from snake_sim.cpp_bindings.area_check import AreaChecker
from snake_sim.cpp_bindings.utils import get_visitable_tiles


area_checkers: Dict[int, AreaChecker] = {}


def get_area_checkers(
        snake_values: Dict[int, Dict[str, int]], 
        free_value: int, 
        food_value: int, 
        width: int,
        height: int,
    ) -> AreaChecker:
    global area_checkers
    for snake_id, values in snake_values.items():
        if snake_id not in area_checkers:
            area_checkers[snake_id] = AreaChecker(
                food_value,
                free_value,
                values['body_value'],
                values['head_value'],
                width,
                height,
            )


def get_best_area_checks(
        area_checkers: Dict[int, AreaChecker],
        state: CompleteStepState, 
        s_map: np.ndarray
    ) -> dict[int, AreaCheckResult]:
    """ Check if each snake can access the area around it. """
    best_area_checks: dict[int, AreaCheckResult] = {}
    for snake_id, body in state.snake_bodies.items():
        if not state.snake_alive.get(snake_id, False):
            continue
        head_coord = body[0]
        visitable_tiles = get_visitable_tiles(
            s_map,
            state.env_meta_data.width,
            state.env_meta_data.height,
            head_coord,
            [state.env_meta_data.free_value, state.env_meta_data.food_value]
        )
        for tile in visitable_tiles:
            target_margin = max(10, math.ceil(0.06 * len(body)))
            result = area_checkers[snake_id].area_check(
                s_map,
                list(body),
                tile,
                target_margin=target_margin,
                food_check=False,
                complete_area=False,
                exhaustive=False
            )
            current_best = best_area_checks.get(snake_id)
            if current_best is None or current_best.margin < result['margin']:
                best_area_checks[snake_id] = AreaCheckResult(**result)

    return best_area_checks


def compute_rewards(state_map1: tuple[CompleteStepState, np.ndarray],
                state_map2: tuple[CompleteStepState, np.ndarray],
                snake_ids: set[int]) -> dict[int, float]:
    """ Computes rewards for each snake between two states.
    """
    state1, map1 = state_map1
    state2, map2 = state_map2
    rewards = {}
    get_area_checkers(
        snake_values=state1.env_meta_data.snake_values,
        free_value=state1.env_meta_data.free_value,
        food_value=state1.env_meta_data.food_value,
        width=state1.env_meta_data.width,
        height=state1.env_meta_data.height,
    )
    best_area_checks = get_best_area_checks(area_checkers, state2, map2)
    if debug.is_debug_active():
        for area_snake_id, area_check in best_area_checks.items():
            print(f"Area check for snake {area_snake_id}: {area_check}")
            
    for s_id in snake_ids:
        if s_id not in state_map1[0].snake_bodies or s_id not in state_map2[0].snake_bodies:
            raise ValueError(f"Snake id {s_id} not found in one of the states.")
        food_dist1 = distance_to_tile_with_value(
            map1,
            state1.env_meta_data.width,
            state1.env_meta_data.height,
            state1.snake_bodies[s_id][0],
            state1.env_meta_data.food_value,
            [state1.env_meta_data.free_value, state1.env_meta_data.food_value]
        )
        food_dist2 = distance_to_tile_with_value(
            map2,
            state2.env_meta_data.width,
            state2.env_meta_data.height,
            state2.snake_bodies[s_id][0],
            state2.env_meta_data.food_value,
            [state2.env_meta_data.free_value, state2.env_meta_data.food_value]
        )
        did_eat = state2.snake_bodies[s_id][0] in state1.food
        if set(state1.food) != set(state2.food) or did_eat:
            # if snake ate or food changed, distances are unreliable
            food_dist1 = None
            food_dist2 = None
        survival_chance_reward = _survival_chance_reward(best_area_checks.get(s_id))
        food_reward = _food_approach_reward(food_dist1, food_dist2)
        did_eat_reward = _food_eat_reward(did_eat)
        length_reward = _length_reward(
            len(state1.snake_bodies[s_id]),
            len(state2.snake_bodies[s_id]),
            state2.snake_alive.get(s_id, False)
        )
        survival_reward = _survival_reward(
            state2.snake_alive.get(s_id, False),
            len(state2.snake_bodies[s_id]) if s_id in state2.snake_bodies else 2
        )
        # Count total snakes that have died so far (cumulative)
        total_dead = sum(1 for sid in snake_ids if not state2.snake_alive.get(sid, False))
        dominance_reward = _dominance_reward(state2.snake_alive.get(s_id, False), total_dead)
        total_reward = sum(
            (
                length_reward,
                survival_reward,
                food_reward,
                did_eat_reward,
                survival_chance_reward,
                dominance_reward
            )
        )
        # Scale overall reward signal to increase training SNR
        total_reward = float(total_reward) * 10.0
        rewards[s_id] = total_reward

        if debug.is_debug_active():
            print(
                f"🎯 Rewards for snake {s_id}: length={length_reward:.2f}, survival={survival_reward:.2f}, "
                f"food_approach={food_reward:.2f}, food_eat={did_eat_reward:.2f}, "
                f"survival_chance={survival_chance_reward:.2f}, dominance={dominance_reward:.2f}, total={total_reward:.2f}"
            )
    
    return rewards


def _food_approach_reward(dist1: float | None, dist2: float | None) -> float:
    if dist1 is None or dist2 is None:
        return 0.0
    if dist2 < dist1:
        return 0.01  # Scaled down from 0.1
    elif dist2 > dist1:
        return -0.01  # Scaled down from -0.1
    else:
        return 0.0  # no change


def _food_eat_reward(ate_food: bool) -> float:
    if ate_food:
        return 1.0  # Scaled down from 10.0
    return 0.0  # did not eat food


def _length_reward(len1: int, len2: int, still_alive: bool) -> float:
    if len2 > len1 and still_alive:
        # Reward scales with current length - bigger snakes get more reward for eating
        length_multiplier = 1.0 + (len2 - 2) * 0.5
        return 0.5 * length_multiplier  # Scaled down from 5.0
    elif len2 < len1:
        return -0.1  # Scaled down from -1.0
    else:
        return 0.0  # no change


def _survival_chance_reward(area_check: AreaCheckResult) -> float:
    if area_check is not None and area_check.margin >= 0:
        return 0.0  # ADDED small bonus for safe positioning
    else:
        return -0.5  # Scaled down from -5.0


def _survival_reward(still_alive: bool, current_length: int = 2) -> float:
    if not still_alive:
        # Fixed death penalty - prevents "die early before penalty grows" incentive
        return -2.0  # Scaled down from -20.0
    else:
        # Small reward scaling with length - bigger snakes get more for surviving
        # This encourages growth without making looping optimal
        return 0.001 * max(0, current_length - 2)  # Scaled down from 0.01


def _dominance_reward(still_alive: bool, total_dead: int) -> float:
    if still_alive:
        # Cumulative reward for surviving when others have died - encourages aggression and long survival
        return 0.05 * total_dead  # Scaled down from 0.5
    else:
        return 0.0

