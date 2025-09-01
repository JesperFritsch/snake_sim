from typing import Dict, Tuple, Union
import numpy as np
import math
from functools import wraps

import snake_sim.debugging as debug

from snake_sim.environment.interfaces.snake_strategy_interface import ISnakeStrategy
from snake_sim.environment.interfaces.snake_strategy_user_interface import ISnakeStrategyUser
from snake_sim.snakes.strategies.strategy_snake_mixin import StrategySnakeMixin
from snake_sim.environment.types import Coord
from snake_sim.snakes.snake import Snake
from snake_sim.utils import print_map

from snake_sim.cpp_bindings.area_check import AreaChecker
from snake_sim.cpp_bindings.utils import get_visitable_tiles


# debug.activate_debug()
# debug.enable_debug_for("SurvivorSnake")
# debug.enable_debug_for("_next_step")


class SurvivorSnake(Snake, StrategySnakeMixin):
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
        self._current_direction = self.coord - self.body_coords[1] if len(self.body_coords) > 1 else None
        debug.debug_print(f"current_tile: {self.coord}, current_direction: {self._current_direction}")
        strategy_tile = self._get_strategy_tile()
        visitable_tiles_here = self._visitable_tiles(self.map, self.coord)
        visitable_tiles_here.sort(
            key=lambda t: (
                0 if strategy_tile == t else
                1 if self._current_direction == t else
                2
            ), 
        ) # prioritize strategy tile if it is visitable
        for tile in visitable_tiles_here:
            debug.debug_print(f"checking tile {tile}")
            tile_coord = Coord(*tile)
            if self._is_move_safe(tile_coord):
                debug.debug_print(f"safe move found: {tile_coord}")
                return tile_coord
        # no safe move found, return None

    def _init_area_checker(self):
        self.area_checker = AreaChecker(
            self.env_init_data.food_value,
            self.env_init_data.free_value,
            self.body_value,
            self.head_value,
            self.env_init_data.width,
            self.env_init_data.height)

    def _is_margin_safe(self, area_check_result) -> bool:
        total_steps = area_check_result['total_steps']
        if total_steps == 0:
            return False
        margin = area_check_result['margin']
        return (margin / total_steps) >= self.SAFE_MARGIN_FACTOR

    def _is_move_safe(self, next_head: Coord) -> bool:
        target_margin = math.ceil(self.length / 10) # without a higher target margin the area check will early exit at (margin > food_count)
        visitable_tiles_after_move = self._visitable_tiles(self.map, next_head)
        if len(visitable_tiles_after_move) == 0:
            return False
        current_head = self.coord
        current_tail = self.body_coords[-1]
        map_copy = self._apply_snake_step(self._current_map_copy, current_head, current_tail, next_head)
        # map_copy is a reference to self._current_map_copy
        for tile in visitable_tiles_after_move:
            area_check_result = self._area_check_wrapper(map_copy, self.body_coords, tile, target_margin=target_margin)
            debug.debug_print(f"area_check_result for move to {next_head} with head at {tile}: {area_check_result}")
            if self._is_margin_safe(area_check_result):
                return True
        self._revert_snake_step(map_copy, current_head, current_tail, next_head)
        return False

    def _area_check_wrapper(self, s_map, body_coords, start_coord, target_margin=0, max_food=0, food_check=False, exhaustive=False):
        result = self.area_checker.area_check(s_map, list(body_coords), start_coord, target_margin, max_food, food_check, exhaustive)
        return result

    def _apply_snake_step(self, s_map, current_head: Coord, current_tail: Coord, next_head: Coord) -> np.ndarray:
        s_map[current_tail.y, current_tail.x] = self.env_init_data.free_value
        s_map[current_head.y, current_head.x] = self.body_value
        s_map[next_head.y, next_head.x] = self.head_value
        return s_map

    def _revert_snake_step(self, s_map, old_head: Coord, old_tail: Coord, current_head: Coord) -> np.ndarray:
        s_map[old_tail.y, old_tail.x] = self.body_value
        s_map[old_head.y, old_head.x] = self.head_value
        s_map[current_head.y, current_head.x] = self.env_init_data.free_value
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
        self._current_map_copy = self.map.copy()

    def _post_step_calc(self):
        pass