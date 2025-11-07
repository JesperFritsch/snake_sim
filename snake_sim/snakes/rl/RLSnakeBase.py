from typing import Dict, Tuple, Union, List
import numpy as np
import math
from functools import wraps

import snake_sim.debugging as debug

from snake_sim.environment.types import Coord
from snake_sim.snakes.snake_base import SnakeBase
from snake_sim.cpp_bindings.area_check import AreaChecker

class RLSnakeBase(SnakeBase):
    """ Base class for RL controlled snakes. """

    def __init__(self):
        super().__init__()
        self._area_checker = None # type AreaChecker, will be initialized in set_init_data
        self._current_direction: Coord = None
        self._current_map_copy: np.ndarray = None

    def set_init_data(self, env_meta_data):
        super().set_init_data(env_meta_data)
        self._init_area_checker()
