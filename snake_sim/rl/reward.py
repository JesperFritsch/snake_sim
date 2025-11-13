
import numpy as np

from snake_sim.map_utils.general import print_map
from snake_sim.environment.types import CompleteStepState
from snake_sim.cpp_bindings.utils import distance_to_tile_with_value


def compute_rewards(state_map1: tuple[CompleteStepState, np.ndarray],
                state_map2: tuple[CompleteStepState, np.ndarray],
                snake_ids: set[int]) -> dict[int, float]:
    """ Computes rewards for each snake between two states.
    """
    state1, map1 = state_map1
    state2, map2 = state_map2
    rewards = {}
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
        total_reward = length_reward + survival_reward + food_reward + did_eat_reward
        rewards[s_id] = total_reward
    return rewards


def _food_approach_reward(dist1: float | None, dist2: float | None) -> float:
    if dist1 is None or dist2 is None:
        return 0.0
    if dist2 < dist1:
        return 0.3  # Increased from 0.1 - approaching food
    elif dist2 > dist1:
        return -0.1  # moving away from food
    else:
        return 0.0  # no change


def _food_eat_reward(ate_food: bool) -> float:
    if ate_food:
        return 1.0  # ate food
    return 0.0  # did not eat food


def _length_reward(len1: int, len2: int, still_alive: bool) -> float:
    if len2 > len1 and still_alive:
        # Reward scales with current length - bigger snakes get more reward for eating
        length_multiplier = 1.0 + (len2 - 2) * 0.1  # Bonus increases with size
        return 2.0 * length_multiplier  # Base eating reward increased
    elif len2 < len1:
        return -1.0  # shrank (should not happen normally)
    else:
        return 0.0  # no change


def _survival_reward(still_alive: bool, current_length: int = 2) -> float:
    if not still_alive:
        # No explicit death penalty - let episode termination be the penalty
        # The lost opportunity for future rewards is the natural consequence
        return 0.0
    else:
        return 0.05  # Survival bonus per step

