
import math
import numpy as np
from typing import Dict

import snake_sim.debugging as debug
from snake_sim.environment.types import CompleteStepState
from snake_sim.environment.types import AreaCheckResult
from snake_sim.cpp_bindings.area_check import AreaChecker
from snake_sim.cpp_bindings.utils import (
    get_visitable_tiles, 
    area_boundary_tiles, 
    distance_to_tile_with_value, 
    distance_to_coord
)


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
        if best_area_checks.get(snake_id) is None:
            # If no visitable tiles, do a check on the head tile to get the margin (which will be negative)
            best_area_checks[snake_id] = AreaCheckResult(
                False,
                0,
                0,
                0,
                False,
                -1,
                0,
            )

    return best_area_checks


def recently_trapped(
    area_checks1: dict[int, AreaCheckResult],
    area_checks2: dict[int, AreaCheckResult]
) -> set[int]:
    """ Detect snakes that have recently become trapped.
    
    A snake is considered recently trapped if its area check margin has dropped
    from non-negative to negative between two checks.
    """
    trapped_snakes = set()
    for snake_id, check2 in area_checks2.items():
        check1 = area_checks1.get(snake_id)
        if check1 is not None and check2 is not None:
            if ((check2.margin < check1.margin * 0.1) # if margin dropped significantly check if some snake contributed to the trapping
                or (check1.margin >= 0 and check2.margin < 0)):
                trapped_snakes.add(snake_id)
    debug.debug_print(f"Trapped snakes: {trapped_snakes}")
    return trapped_snakes


def assign_trapping_credit(
    trapped_snakes: set[int],
    state: CompleteStepState,
    s_map: np.ndarray,
    snake_ids: set[int]
) -> dict[int, bool]:

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
    contributors: dict[int, bool] = {}
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
                [state.env_meta_data.free_value, state.env_meta_data.food_value],
                target_is_visitable=False
            ) for coord in trapping_visitable_tiles]
            debug.debug_print(f"Distances from snake {trapping_id} to trapped snake {trapped_id} head: {distances_to_trapped_head}")
            if head_value in boundary_tiles and any(d >= 0 for d in distances_to_trapped_head):
                # If the trapping snake's head is on the boundary of the trapped area and can reach the trapped snake's head, assign credit.
                # but if the trapping snakes head in in the same area as the trapped snake, it shouldn't get credit (prevents rewarding snakes for being near the trapped snake if they are also trapped)
                contributors[trapping_id] = True
                break
    return contributors


def compute_rewards(state_map1: tuple[CompleteStepState, np.ndarray],
                state_map2: tuple[CompleteStepState, np.ndarray],
                snake_ids: set[int]) -> tuple[dict[int, float], dict[int, dict[str, float]]]:
    """ Computes rewards for each snake between two states.
    """
    state1, map1 = state_map1
    state2, map2 = state_map2
    rewards = {}
    info = {}
    if not len(area_checkers):
        get_area_checkers(
            snake_values=state1.env_meta_data.snake_values,
            free_value=state1.env_meta_data.free_value,
            food_value=state1.env_meta_data.food_value,
            width=state1.env_meta_data.width,
            height=state1.env_meta_data.height,
        )
    ids_to_check = set([id for id, alive in state1.snake_alive.items() if alive])
    best_area_checks_s1 = get_best_area_checks(area_checkers, state1, map1, ids_to_check)
    best_area_checks_s2 = get_best_area_checks(area_checkers, state2, map2, ids_to_check)

    trapped_snakes = recently_trapped(best_area_checks_s1, best_area_checks_s2)
    trapping_contributors = assign_trapping_credit(trapped_snakes, state2, map2, snake_ids)

    if debug.is_debug_active():
        for area_snake_id, area_check in best_area_checks_s2.items():
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
        survival_chance_reward = _survival_chance_reward(best_area_checks_s2.get(s_id))
        food_reward = _food_approach_reward(food_dist1, food_dist2)
        did_eat_reward = _food_eat_reward(did_eat)
        survival_reward = _survival_reward(state2.snake_alive.get(s_id, False))
        trapping_reward_value = trapping_reward(
            trapping_contributors.get(s_id, False)
        )
        total_reward = sum(
            (
                survival_reward,
                food_reward,
                did_eat_reward,
                last_standing_reward_value,
                survival_chance_reward,
                trapping_reward_value,
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
            'total_reward': total_reward,
        }

        if debug.is_debug_active():
            print(
                f"🎯 Rewards for snake {s_id}:, survival={survival_reward:.2f}, "
                f"food_approach={food_reward:.2f}, food_eat={did_eat_reward:.2f}, "
                f"survival_chance={survival_chance_reward:.2f}, trapping={trapping_reward_value:.2f}, total={total_reward:.2f}"
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

