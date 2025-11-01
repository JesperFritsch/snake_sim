from __future__ import annotations

from snake_sim.environment.types import Coord
from snake_sim.environment.interfaces.snake_strategy_interface import ISnakeStrategy

import snake_sim.debugging as debug
from snake_sim.cpp_bindings.utils import get_dir_to_tile, get_visitable_tiles, can_make_area_inaccessible
from snake_sim.cpp_bindings.area_check import AreaChecker


class DeepStrategy(ISnakeStrategy):
    """ A simple strategy that tries to get to the closest food """

    def __init__(self):
        super().__init__()
        self._area_checker = None

    