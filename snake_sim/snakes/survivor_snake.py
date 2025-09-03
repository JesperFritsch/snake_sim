from typing import Dict, Tuple, Union, List
import numpy as np
import math
from functools import wraps

import snake_sim.debugging as debug

from snake_sim.environment.interfaces.snake_strategy_interface import ISnakeStrategy
from snake_sim.environment.interfaces.strategy_snake_interface import IStrategySnake
from snake_sim.environment.types import Coord
from snake_sim.snakes.snake_base import SnakeBase
from snake_sim.utils import print_map, distance

from snake_sim.cpp_bindings.area_check import AreaChecker
from snake_sim.cpp_bindings.utils import get_visitable_tiles


# debug.activate_debug()
# debug.enable_debug_for("SurvivorSnake")
# debug.enable_debug_for("FoodSeeker")
# debug.enable_debug_for("_next_step")


class SurvivorSnake(IStrategySnake, SnakeBase):
    """ This is a snake that without a strategy (ISnakeStrategy) only tries to survive.
    It will just continue in its current direction until it needs to make a turn to avoid dying.
    If it is provided with a strategy, it will try to follow that strategy as long as it does not lead to death.
    """

    SAFE_MARGIN_FACTOR = 0.035 # (margin / total_steps) >= SAFE_MARGIN_FACTOR -> considered safe

    def __init__(self):
        super().__init__()
        self._area_checker = None # type AreaChecker, will be initialized in set_init_data
        self._current_direction: Coord = None
        self._current_map_copy: np.ndarray = None
        self._area_check_cache: Dict[Coord, List[Dict]] = {}
        self._current_visitable_tiles: List[Coord] = []
        self._current_strategy_tile: Coord = None

    def set_init_data(self, env_init_data):
        super().set_init_data(env_init_data)
        self._init_area_checker()

    def _step_calc_wrapper(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            self._pre_step_calc()
            result = func(self, *args, **kwargs)
            self._post_step_calc()
            return result
        return wrapper

    # from abstract class Snake
    @_step_calc_wrapper
    def _next_step(self) -> Coord:
        debug.debug_print(f"current_tile: {self._head_coord}, current_direction: {self._current_direction}")
        for tile in self._current_visitable_tiles:
            debug.debug_print(f"checking tile {tile}")
            tile_coord = Coord(*tile)
            if self._is_move_safe(tile_coord):
                debug.debug_print(f"safe move found: {tile_coord}")
                return tile_coord
        best_option = self._get_best_opiton_from_cache()
        if best_option is not None:
            debug.debug_print(f"no safe move found, using best option from cache: {best_option}")
            return best_option
        debug.debug_print("no safe move found, no best option in cache")
        immediate_best = self._immediate_steps_checks(self._current_visitable_tiles)
        if immediate_best is not None:
            debug.debug_print(f"no safe move found, using immediate best option: {immediate_best}")
            return immediate_best
        debug.debug_print("no safe move found, no immediate best option")
        return None
        
    def _init_area_checker(self):
        self._area_checker = AreaChecker(
            self._env_init_data.food_value,
            self._env_init_data.free_value,
            self._body_value,
            self._head_value,
            self._env_init_data.width,
            self._env_init_data.height)

    def _is_margin_safe(self, area_check_result) -> bool:
        total_steps = area_check_result['total_steps']
        if total_steps == 0:
            return False
        food_count = area_check_result['food_count']
        margin = area_check_result['margin']
        return (margin / total_steps) >= self.SAFE_MARGIN_FACTOR and margin >= food_count 

    def _is_move_safe(self, next_head: Coord) -> bool:
        target_margin = math.ceil(self._length / 10) # without a higher target margin the area check will early exit at (margin > food_count)
        visitable_tiles_after_move = self._visitable_tiles(self._map, next_head)
        if len(visitable_tiles_after_move) == 0:
            return False
        current_head = self._head_coord
        current_tail = self._body_coords[-1]
        map_copy = self._apply_snake_step(self._current_map_copy, current_head, current_tail, next_head)
        # map_copy is a reference to self._current_map_copy
        for tile in visitable_tiles_after_move:
            area_check_result = self._area_check_wrapper(map_copy, self._body_coords, tile, target_margin=target_margin)
            self._area_check_cache.setdefault(next_head, []).append(area_check_result)
            debug.debug_print(f"area_check_result for move to {next_head} with head at {tile}: {area_check_result}")
            if self._is_margin_safe(area_check_result):
                return True
        self._revert_snake_step(map_copy, current_head, current_tail, next_head)
        return False

    def _area_check_wrapper(self, s_map, body_coords, start_coord, target_margin=0, food_check=False, complete_area=False, exhaustive=False):
        result = self._area_checker.area_check(s_map, list(body_coords), start_coord, target_margin, food_check, complete_area, exhaustive)
        return result

    def _get_best_opiton_from_cache(self) -> Union[Coord, None]:
        best_option = max(
            ((tile, results) for tile, results in self._area_check_cache.items()), 
            key=lambda item: (
                max(
                    (res['margin'] for res in item[1]),
                    default=-1
                )
            ),
            default=(None, None)
        )
        if best_option[0] is None:
            return None
        closest_to_strat_tile = min(
            self._current_visitable_tiles, 
            key=lambda tile: distance(self._current_strategy_tile, tile), default=None
        )
        if closest_to_strat_tile is not None:
            cached_area_checks = self._area_check_cache.get(closest_to_strat_tile, [])
            if any(map(self._is_margin_safe, cached_area_checks)):
                return Coord(*closest_to_strat_tile)

        return Coord(*best_option[0])

    def _immediate_steps_checks(self, visitable_tiles: List[Coord]) -> Coord:
        area_checks = {coord: self._area_check_wrapper(self._map, self._body_coords, coord) for coord in visitable_tiles}
        debug.debug_print(f"immediate area_checks: {area_checks}")
        best_tile = max(area_checks, key=lambda c: area_checks[c]['margin'], default=None)
        if best_tile is None:
            return None
        return Coord(*best_tile)

    def _apply_snake_step(self, s_map, current_head: Coord, current_tail: Coord, next_head: Coord) -> np.ndarray:
        s_map[current_tail.y, current_tail.x] = self._env_init_data.free_value
        s_map[current_head.y, current_head.x] = self._body_value
        s_map[next_head.y, next_head.x] = self._head_value
        return s_map

    def _revert_snake_step(self, s_map, old_head: Coord, old_tail: Coord, current_head: Coord) -> np.ndarray:
        s_map[old_tail.y, old_tail.x] = self._body_value
        s_map[old_head.y, old_head.x] = self._head_value
        s_map[current_head.y, current_head.x] = self._env_init_data.free_value
        return s_map

    def _visitable_tiles(self, s_map, coord):
        return list(
            get_visitable_tiles(
                s_map,
                self.get_env_init_data().width,
                self.get_env_init_data().height,
                coord,
                [self.get_env_init_data().free_value, self.get_env_init_data().food_value]
            )
        )

    def _pre_step_calc(self):
        self._current_map_copy = self._map.copy()
        self._area_check_cache = {}
        self._current_direction = self._head_coord - self._body_coords[1] if len(self._body_coords) > 1 else None
        self._current_strategy_tile = self._get_strategy_tile()
        self._current_visitable_tiles = self._visitable_tiles(self._map, self._head_coord)
        self._current_visitable_tiles.sort(
            key=lambda t: (
                0 if self._current_strategy_tile == t else
                1 if self._current_direction == t else
                2
            ), 
        ) # prioritize strategy tile if it is visitable

    def _post_step_calc(self):
        pass