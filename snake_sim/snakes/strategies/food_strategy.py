from __future__ import annotations

from snake_sim.environment.types import Coord
from snake_sim.environment.interfaces.snake_strategy_interface import ISnakeStrategy
from snake_sim.utils import get_coord_parity

import snake_sim.debugging as debug
from snake_sim.cpp_bindings.utils import get_dir_to_tile, get_visitable_tiles, can_make_area_inaccessible
from snake_sim.cpp_bindings.area_check import AreaChecker


class FoodSeeker(ISnakeStrategy):
    """ A simple strategy that tries to get to the closest food """

    def __init__(self):
        super().__init__()
        self._area_checker = None

    def get_wanted_tile(self) -> Coord:
        if self._area_checker is None:
            self._init_area_checker()
        food_dir_tile = self._get_food_dir_tile()
        if food_dir_tile is None:
            return None
        if self.can_close_area():
            food_map = self._get_future_available_food_map()
        else:
            food_map = {}
        debug.debug_print(f"food_map: {food_map}")
        if not food_map:
            return food_dir_tile
        best_food_tile = self._get_best_food_option(food_map, food_dir_tile)
        debug.debug_print(f"food_dir_tile: {food_dir_tile}, best_food_tile: {best_food_tile}, food_map: {food_map}")
        return Coord(*best_food_tile)
        
    def _get_food_dir_tile(self) -> Coord:
        env_init_data = self._snake.get_env_init_data()
        s_map = self._snake.get_map()
        coord = self._snake.get_head_coord()
        dir_tuple = get_dir_to_tile(
            s_map,
            env_init_data.width,
            env_init_data.height,
            coord,
            env_init_data.food_value,
            [env_init_data.free_value, env_init_data.food_value],
            clockwise=get_coord_parity(coord) or True
        )
        if dir_tuple == (0,0):
            return None
        return self._snake.get_head_coord() + Coord(*dir_tuple)
    
    def _get_future_available_food_map(self):
        s_map = self._snake.get_map()
        env_init_data = self._snake.get_env_init_data()
        head_coord = self._snake.get_head_coord()
        visitable_tiles = get_visitable_tiles(
            s_map,
            env_init_data.width,
            env_init_data.height,
            head_coord,
            [env_init_data.free_value, env_init_data.food_value]
        )
        debug.debug_print(f"visitable_tiles: {visitable_tiles}")
        food_map = {
            Coord(*coord): a["food_count"] 
            if a["margin"] >= a["food_count"] else 0
            for coord, a in
            [(coord, self._area_check_wrapper(coord)) for coord in visitable_tiles]
        }

        # combine_food = all([a['margin'] >= a['food_count'] and a["food_count"] > 0 for a in all_checks]) or self.length < 15
        return food_map
        
    def _get_best_food_option(self, food_map: dict, food_tile: Coord) -> Coord:
        food_dir_value = food_map.get(food_tile, 0)
        best_food_tile = max(food_map, key=food_map.get, default=None)
        best_food_value = food_map.get(best_food_tile, 0)
        debug.debug_print(f"food_dir_value: {food_dir_value}, best_food_value: {best_food_value}")
        debug.debug_print(f"food_tile: {food_tile}, best_food_tile: {best_food_tile}")
        if food_dir_value >= best_food_value:
            debug.debug_print("Choosing food_dir_tile")
            return food_tile
        debug.debug_print("Choosing best_food_tile")
        return best_food_tile


    def _area_check_wrapper(self, start_coord: Coord=None, target_margin=0, food_check=False, complete_area=False, exhaustive=False):
        s_map = self._snake.get_map().copy()
        body_coords = self._snake.get_body_coords()
        # block the head coords so that area check does not count it as free space
        # s_map[start_coord.y, start_coord.x] = self._snake.get_env_init_data().blocked_value
        result = self._area_checker.area_check(
            s_map,
            list(body_coords),
            start_coord,
            target_margin=0,
            food_check=food_check,
            complete_area=complete_area,
            exhaustive=exhaustive
        )
        debug.debug_print(f"Area check for {start_coord}: {result}")
        return result

    def can_close_area(self) -> bool:
        init_data = self._snake.get_env_init_data()
        if self._snake._length < 10:
            return False
        return can_make_area_inaccessible(
            self._snake.get_map(),
            init_data.width,
            init_data.height,
            init_data.free_value,
            self._snake.get_head_coord(),
            self._snake.get_body_coords()[1]
        )

    def _init_area_checker(self):
        env_init_data = self._snake.get_env_init_data()
        head_value, body_value = self._snake.get_self_map_values()
        self._area_checker = AreaChecker(
            env_init_data.food_value,
            env_init_data.free_value,
            body_value,
            head_value,
            env_init_data.width,
            env_init_data.height
        )