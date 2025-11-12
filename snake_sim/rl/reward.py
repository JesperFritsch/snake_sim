
import numpy as np

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
        food_reward = _food_approach_reward(food_dist1, food_dist2)
        length_reward = _length_reward(
            len(state1.snake_bodies[s_id]),
            len(state2.snake_bodies[s_id])
        )
        survival_reward = _survival_reward(
            state1.snake_bodies[s_id][0] == state2.snake_bodies[s_id][0]
        )
        total_reward = food_reward + length_reward + survival_reward
        rewards[s_id] = total_reward
    return rewards


def _food_approach_reward(dist1: float | None, dist2: float | None) -> float:
    if dist1 is None or dist2 is None:
        return 0.0
    if dist2 < dist1:
        return 0.1  # approaching food
    elif dist2 > dist1:
        return -0.1  # moving away from food
    else:
        return 0.0  # no change
    

def _length_reward(len1: int, len2: int) -> float:
    if len2 > len1:
        return 1.0  # grew
    elif len2 < len1:
        return -1.0  # shrank (should not happen normally)
    else:
        return 0.0  # no change


def _survival_reward(still_alive: bool) -> float:
    if not still_alive:
        return -1.0  # died
    else:
        return 0.01  # survived this step
