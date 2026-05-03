
import math
import numpy as np
from typing import Dict

import snake_sim.debugging as debug
from snake_sim.environment.types import AreaCheckResult, CompleteStepState
from snake_sim.analyze.scripts.run_analyzer import (
    find_traps, 
    get_area_checkers, 
    get_best_area_checks
)
from snake_sim.cpp_bindings.utils import (
    distance_to_tile_with_value, 
    voronoi_maps
)


def get_voronoi_results(state: CompleteStepState, map: np.ndarray) -> Dict[int, int]:
    """Get Voronoi map results for each snake."""
    alive_snake_coords = {
        s_id: state.snake_bodies[s_id][0]  # head coordinate
        for s_id, alive in state.snake_alive.items() if alive
    }
    voronoi = voronoi_maps(
        map,
        state.env_meta_data.width,
        state.env_meta_data.height,
        state.env_meta_data.free_value,
        alive_snake_coords,
        np.array(map.shape, dtype=np.int32),
        np.array(map.shape, dtype=np.int32)
    )
    return voronoi

def compute_rewards(state_map1: tuple[CompleteStepState, np.ndarray],
                state_map2: tuple[CompleteStepState, np.ndarray],
                snake_ids: set[int]) -> tuple[dict[int, float], dict[int, dict[str, float]]]:
    """ Computes rewards for each snake between two states.
    """
    state1, map1 = state_map1
    state2, map2 = state_map2
    rewards = {}
    info = {}
    area_checkers = get_area_checkers(
        snake_values=state1.env_meta_data.snake_values,
        free_value=state1.env_meta_data.free_value,
        food_value=state1.env_meta_data.food_value,
        width=state1.env_meta_data.width,
        height=state1.env_meta_data.height,
    )
    ids_to_check = set([id for id, alive in state1.snake_alive.items() if alive])
    best_area_checks = get_best_area_checks(area_checkers, state2, map2, ids_to_check)

    traps_set = find_traps(state1, state2, map1, map2, snake_ids) if len(snake_ids) > 1 else set()

    curr_voronoi = get_voronoi_results(state2, map2)
    prev_voronoi = get_voronoi_results(state1, map1)

    if debug.is_debug_active():
        print("Best area checks:")
        for snake_id, area_check in best_area_checks.items():
            print(f"Snake {snake_id}: {area_check}")
        print("Traps found:")
        for trap in traps_set:
            print(trap)

    if debug.is_debug_active():
        for area_snake_id, area_check in best_area_checks.items():
            print(f"Area check for snake {area_snake_id}: {area_check}")
            
    for s_id in snake_ids:
        if s_id not in state_map1[0].snake_bodies or s_id not in state_map2[0].snake_bodies:
            raise ValueError(f"Snake id {s_id} not found in one of the states.")
        if not state_map1[0].snake_alive.get(s_id, False):
            # If the snake is dead in both states, skip reward calculation (no change)
            rewards[s_id] = 0.0
            info[s_id] = {
                'survival_reward': 0.0,
                'food_approach_reward': 0.0,
                'food_eat_reward': 0.0,
                'survival_chance_reward': 0.0,
                'trapping_reward': 0.0,
                'total_reward': 0.0,
            }
            continue

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
        is_last_standing = all(
            not alive for other_id, alive in state2.snake_alive.items() if other_id != s_id
        ) and False
        last_standing_reward_value = last_standing_reward(is_last_standing)
        survival_chance_reward = _survival_chance_reward(best_area_checks.get(s_id))
        food_reward = _food_approach_reward(food_dist1, food_dist2)
        did_eat_reward = _food_eat_reward(did_eat)
        survival_reward = _survival_reward(state2.snake_alive.get(s_id, False))
        trapping_reward_value = trapping_reward(
            any(s_id in trap.trapping_ids for trap in traps_set)
        )
        voronoi_reward = _voronoi_control_reward(prev_voronoi, curr_voronoi, s_id)
        total_reward = sum(
            (
                survival_reward,
                food_reward,
                did_eat_reward,
                last_standing_reward_value,
                survival_chance_reward,
                trapping_reward_value,
                voronoi_reward
            )
        )
        # Scale overall reward signal to increase training SNR
        rewards[s_id] = total_reward

        info[s_id] = {
            'survival_reward': survival_reward,
            'food_approach_reward': food_reward,
            'food_eat_reward': did_eat_reward,
            'survival_chance_reward': survival_chance_reward,
            'trapping_reward': trapping_reward_value,
            'voronoi_reward': voronoi_reward,
            'total_reward': total_reward,
        }

        if debug.is_debug_active():
            print(
                f"🎯 Rewards for snake {s_id}:, survival={survival_reward:.2f}, "
                f"food_approach={food_reward:.2f}, food_eat={did_eat_reward:.2f}, "
                f"survival_chance={survival_chance_reward:.2f}, trapping={trapping_reward_value:.2f}, voronoi={voronoi_reward:.2f}, total={total_reward:.2f}"
            )
    
    return rewards, info


def _food_approach_reward(dist1: float | None, dist2: float | None) -> float:
    """Reward for moving toward food. 
    
    CRITICAL: This is now the PRIMARY reward signal to train food-seeking.
    Stronger signal (1.0 per step) so food-seeking > wall avoidance learned behavior.
    """
    if dist1 is None or dist2 is None:
        return 0.0
    if dist2 < dist1:
        return 0.05  # ← INCREASED: was 0.05. Strong signal to seek food!
    elif dist2 > dist1:
        return -0.1  # Penalty for moving away from food
    else:
        return -0.01  # no change


def _food_eat_reward(ate_food: bool) -> float:
    """Eating food is the primary success signal."""
    if ate_food:
        return 3.0  # ← RESTORED: was 1.0. This is the goal!
    return 0.0  # did not eat food


def _survival_chance_reward(area_check: AreaCheckResult) -> float:
    """Penalize when trapped (no reachable tiles or negative margin)."""
    if area_check is not None and area_check.margin >= 0:
        return 0.0  # ← Small bonus for safe positioning (was 0.0)
    else:
        return -0.5  # Moderate penalty for entering trapped positions


def trapping_reward(is_trapping_contributor: bool) -> float:
    """Reward for contributing to trapping an opponent."""
    if is_trapping_contributor:
        return 5.0  # Reward for contributing to trapping (proportional to food reward)
    return 0.0

def last_standing_reward(is_last_standing: bool) -> float:
    """ Reward per step for being the last snake alive. """
    if is_last_standing:
        return 1.0  # ← RESTORED: was 5.0. Encourage survival to episode end.
    return 0.0

def _survival_reward(still_alive: bool) -> float:
    """Small per-step survival bonus to encourage longer episodes."""
    if not still_alive:
        # Death penalty only if dead
        return -5.0  # Death penalty - meaningful but not overwhelming
    else:
        # Small per-step cost encourages efficient food-seeking over passive survival
        return -0.01

def _voronoi_control_reward(prev_voronoi: Dict[int, int], curr_voronoi: Dict[int, int], s_id: int) -> float:
    """Reward for increasing Voronoi control (number of tiles closer to snake than any other)."""
    prev_self_voronoi = prev_voronoi.get(s_id, 0)
    curr_self_voronoi = curr_voronoi.get(s_id, 0)
    prev_sum_other_voronoi = sum(v for other_id, v in prev_voronoi.items() if other_id != s_id)
    curr_sum_other_voronoi = sum(v for other_id, v in curr_voronoi.items() if other_id != s_id)
    phi_prev = prev_self_voronoi - prev_sum_other_voronoi
    phi_curr = curr_self_voronoi - curr_sum_other_voronoi
    return 0.01 * (phi_curr - phi_prev)  # Scale down to keep as a minor component of the reward signal