import random
import json
from typing import Optional, Set
import importlib.resources as pkg_resources

from snake_sim.utils import get_locations
from snake_sim.environment.types import DotDict, Coord
from snake_sim.environment.interfaces.food_handler_interface import IFoodHandler

with pkg_resources.open_text('snake_sim.config', 'default_config.json') as config_file:
    config = DotDict(json.load(config_file))


class FoodHandler(IFoodHandler):

    def __init__(self, width: int, height: int, max_food: int, decay_count: Optional[int] = None):
        self.width = width
        self.height = height
        self.max_food = max_food
        self.decay_count = decay_count if decay_count else None
        self.locations: Set[Coord] = set()
        self.decay_counters = {}
        self.new_food = []
        self.removed_food = []

    def update(self, s_map):
        self.removed_food = []
        self._remove_old(s_map)
        self._generate_new(s_map)

    def _generate_new(self, s_map):
        empty_tiles = []
        self.new_food = []
        if len(self.locations) >= self.max_food:
            return
        empty_tiles = get_locations(s_map, config.free_value, self.width, self.height)
        for _ in range(self.max_food - len(self.locations)):
            if empty_tiles:
                new_food = random.choice(empty_tiles)
                empty_tiles.remove(new_food)
                self.add_new(new_food)
                self.new_food.append(new_food)
        for location in self.new_food:
            x, y = location
            s_map[y, x] = config.food_value

    def _remove_old(self, s_map):
        if self.decay_count is None:
            return
        for location in set(self.locations):
            self.decay_counters[location] -= 1
            if self.decay_counters[location] <= 0:
                self.remove(location, s_map)

    def add_new(self, coord: Coord):
        self.decay_counters[coord] = self.decay_count
        self.locations.add(coord)

    def remove(self, coord: Coord, s_map):
        if coord in self.locations:
            x, y = coord
            s_map[y, x] = config.free_value
            del self.decay_counters[coord]
            self.locations.remove(coord)
            self.removed_food.append(coord)

    def resize(self, width, height):
        self.width = width
        self.height = height

    def clear(self):
        self.locations.clear()
        self.decay_counters.clear()

    def get_food(self, only_new=False):
        if only_new:
            return self.new_food
        return list(self.locations)
    
    def get_food_diff(self):
        return self.new_food, self.removed_food
