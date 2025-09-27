from typing import Dict, Tuple, Union, List
import numpy as np
import math
from functools import wraps

import snake_sim.debugging as debug

from snake_sim.environment.interfaces.strategy_snake_interface import IStrategySnake
from snake_sim.environment.types import Coord
from snake_sim.snakes.snake_base import SnakeBase
from snake_sim.cpp_bindings.area_check import AreaChecker


# debug.activate_debug()
# debug.enable_debug_for("SurvivorSnake")
# debug.enable_debug_for("FoodSeeker")
# debug.enable_debug_for("_next_step")


class SurvivorSnake(IStrategySnake, SnakeBase):
    """ This is a snake that without a strategy (ISnakeStrategy) only tries to survive.
    It will just continue in its current direction until it needs to make a turn to avoid dying.
    If it is provided with a strategy, it will try to follow that strategy as long as it does not lead to death.
    """

    SAFE_MARGIN_FRAC = 0.06 # (margin / total_steps) >= SAFE_MARGIN_FRAC -> considered safe
    MAX_RECURSE_DEPTH = 1

    def __init__(self):
        super().__init__()
        self._area_checker = None # type AreaChecker, will be initialized in set_init_data
        self._current_direction: Coord = None
        self._current_map_copy: np.ndarray = None
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
        search_first = self._current_strategy_tile if self._current_strategy_tile is not None else self._current_direction
        if search_first is None:
            search_first = Coord(-1, -1)
        debug.debug_print(f"search_first: {search_first}")
        margin_fracs = self._get_margin_fracs(search_first)
        best_option = self._get_best_option(margin_fracs)
        debug.debug_print(f"best_option: {best_option}")
        return best_option
        
    def _get_margin_fracs(self, search_first: Coord) -> Dict[Coord, Dict[int, float]]:
        target_margin = max(10, math.ceil(self.SAFE_MARGIN_FRAC * len(self._body_coords)))
        result = self._area_checker.recurse_area_check(
            self._current_map_copy,
            list(self._body_coords),
            search_first,
            target_margin,
            self.MAX_RECURSE_DEPTH,
            self.SAFE_MARGIN_FRAC
        )
        converted = {Coord(*coord): res for coord, res in result.items()}
        debug.debug_print(f"margin_fracs: {converted}")
        return converted
    
    def _get_best_option(self, margin_fracs: Dict[Coord, Dict[int, float]]) -> Union[Coord, None]:
        max_depth = max([len(depths) for depths in margin_fracs.values()]) if len(margin_fracs) > 0 else 0
        debug.debug_print(f"max_depth found: {max_depth}")
        if not margin_fracs:
            debug.debug_print("no margin fracs found, will die")
            return None
        strat_margin_fracs = margin_fracs[self._current_strategy_tile] if self._current_strategy_tile in margin_fracs else {}
        strat_margin_frac_max_depth = max(strat_margin_fracs.keys()) if len(strat_margin_fracs) > 0 else -1
        strat_margin_frac_at_depth = strat_margin_fracs.get(strat_margin_frac_max_depth, -1)
        debug.debug_print(f"max_depth: {strat_margin_frac_max_depth}, strat_margin_frac_at_depth: {strat_margin_frac_at_depth}")
        if strat_margin_frac_at_depth >= self.SAFE_MARGIN_FRAC:
            debug.debug_print(f"choosing strategy tile {self._current_strategy_tile} with margin frac {strat_margin_frac_at_depth}")
            return self._current_strategy_tile
        else:
            return max(
                margin_fracs,
                key=lambda coord: margin_fracs[coord].get(max(margin_fracs[coord].keys()), -1)
            )

    def _init_area_checker(self):
        self._area_checker = AreaChecker(
            self._env_init_data.food_value,
            self._env_init_data.free_value,
            self._body_value,
            self._head_value,
            self._env_init_data.width,
            self._env_init_data.height)

    def _pre_step_calc(self):
        self._current_map_copy = self._map.copy()
        self._current_direction = self._head_coord - self._body_coords[1] if len(self._body_coords) > 1 else None
        self._current_strategy_tile = self._get_strategy_tile()

    def _post_step_calc(self):
        pass