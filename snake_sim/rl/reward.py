
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
from snake_sim.cpp_bindings.utils import get_visitable_tiles, area_boundary_tiles


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
            if check1.margin >= 0 and check2.margin < 0:
                trapped_snakes.add(snake_id)
    return trapped_snakes


def assign_trapping_credit(
    trapped_snakes: set[int],
    state: CompleteStepState,
    s_map: np.ndarray,
    snake_ids: set[int]
) -> dict[int, bool]:

    if state is None or not trapped_snakes:
        return {}
    
    trapping_area_boundaries: set[int] = set()


    for snake_id in trapped_snakes:
        head_coord = state.snake_bodies[snake_id][0]
        boundary_tiles = area_boundary_tiles(
            s_map,
            state.env_meta_data.width,
            state.env_meta_data.height,
            state.env_meta_data.free_value,
            head_coord
        )
        trapping_area_boundaries.update(boundary_tiles)
    contributors: dict[int, bool] = {}
    for s_id in snake_ids:
        if not state.snake_alive.get(s_id, False):
            continue
        head_value = state.env_meta_data.snake_values[s_id]['head_value']
        if head_value in trapping_area_boundaries:
            contributors[s_id] = True
    return contributors


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
    best_area_checks_s1 = get_best_area_checks(area_checkers, state1, map1)
    best_area_checks_s2 = get_best_area_checks(area_checkers, state2, map2)

    trapped_snakes = recently_trapped(best_area_checks_s1, best_area_checks_s2)
    trapping_contributors = assign_trapping_credit(trapped_snakes, state2, map2, snake_ids)

    if debug.is_debug_active():
        for area_snake_id, area_check in best_area_checks_s2.items():
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
        survival_chance_reward = _survival_chance_reward(best_area_checks_s2.get(s_id))
        food_reward = _food_approach_reward(food_dist1, food_dist2)
        did_eat_reward = _food_eat_reward(did_eat)
        survival_reward = _survival_reward(
            state2.snake_alive.get(s_id, False),
            len(state2.snake_bodies[s_id]) if s_id in state2.snake_bodies else 2
        )
        trapping_reward_value = trapping_reward(
            trapping_contributors.get(s_id, False)
        )
        total_reward = sum(
            (
                survival_reward,
                food_reward,
                did_eat_reward,
                survival_chance_reward,
                trapping_reward_value,
            )
        )
        # Scale overall reward signal to increase training SNR
        rewards[s_id] = total_reward

        if debug.is_debug_active():
            print(
                f"🎯 Rewards for snake {s_id}:, survival={survival_reward:.2f}, "
                f"food_approach={food_reward:.2f}, food_eat={did_eat_reward:.2f}, "
                f"survival_chance={survival_chance_reward:.2f}, trapping={trapping_reward_value:.2f}, total={total_reward:.2f}"
            )
    
    return rewards


def _food_approach_reward(dist1: float | None, dist2: float | None) -> float:
    """Reward for moving toward food. 
    
    CRITICAL: This is now the PRIMARY reward signal to train food-seeking.
    Stronger signal (1.0 per step) so food-seeking > wall avoidance learned behavior.
    """
    if dist1 is None or dist2 is None:
        return 0.0
    if dist2 < dist1:
        return 0.01  # ← INCREASED: was 0.05. Strong signal to seek food!
    elif dist2 > dist1:
        return -0.01  # ← INCREASED penalty: was -0.05. Discourage moving away.
    else:
        return 0.0  # no change


def _food_eat_reward(ate_food: bool) -> float:
    """Eating food is the primary success signal."""
    if ate_food:
        return .5  # ← RESTORED: was 1.0. This is the goal!
    return 0.0  # did not eat food


def _survival_chance_reward(area_check: AreaCheckResult) -> float:
    """Penalize when trapped (no reachable tiles or negative margin)."""
    if area_check is not None and area_check.margin >= 0:
        return 0.0  # ← Small bonus for safe positioning (was 0.0)
    else:
        return -0.5  # ← Moderate penalty: was -0.2. Discourage going into traps.


def trapping_reward(is_trapping_contributor: bool) -> float:
    """Reward for contributing to trapping an opponent."""
    if is_trapping_contributor:
        return 5 # Reward for directly trapping an opponent
    return 0.0


def _survival_reward(still_alive: bool, current_length: int = 2) -> float:
    """Small per-step survival bonus to encourage longer episodes."""
    if not still_alive:
        # Death penalty only if dead
        return -10.0  # Keep penalty for death
    else:
        # Small survival bonus - encourages long episodes needed for food exploration
        return -0.001  # ← RESTORED: was 0 (commented out). 1 cent per step to explore!


def _dominance_reward(still_alive: bool, total_dead: int) -> float:
    """DISABLED: Dominance rewards cause agents to wait passively for opponents to die.
    
    The previous implementation (0.1 * total_dead) incentivizes sitting still:
    - As soon as 1 opponent dies, agent gets +0.1/step bonus forever
    - With 15 snakes, after a few die, agent gets +1.0+/step bonus for doing nothing
    - This is stronger than food-seeking rewards, so agents learn passivity
    
    For now, disable this to force agents to learn food-seeking instead of waiting.
    Once food-seeking is solid, re-enable with careful tuning.
    """
    return 0.0  # Disabled

