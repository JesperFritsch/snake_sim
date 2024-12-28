import random
import json
from typing import Optional
import importlib.resources as pkg_resources

from snake_sim.utils import DotDict, Coord
from snake_sim.environment.interfaces.food_handler_interface import IFoodHandler

with pkg_resources.open_text('snake_sim.config', 'default_config.json') as config_file:
    config = DotDict(json.load(config_file))


class FoodHandler(IFoodHandler):

    def __init__(self, width: int, height: int, max_food: int, decay_count: Optional[int] = None):
        self.width = width
        self.height = height
        self.max_food = max_food
        self.decay_count = decay_count if decay_count else None
        self.locations = set()
        self.decay_counters = {}
        self.newest_food = []

    def update(self, s_map):
        self._remove_old(s_map)
        self._generate_new(s_map)

    def _generate_new(self, s_map):
        empty_tiles = []
        self.newest_food = []
        for y in range(self.height):
            for x in range(self.width):
                if s_map[y, x] == config.free_value:
                    empty_tiles.append((x, y))
        for _ in range(self.max_food - len(self.locations)):
            if empty_tiles:
                new_food = random.choice(empty_tiles)
                empty_tiles.remove(new_food)
                self.add_new(new_food)
                self.newest_food.append(new_food)
        for location in self.newest_food:
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

    def resize(self, width, height):
        self.width = width
        self.height = height

    def clear(self):
        self.locations.clear()
        self.decay_counters.clear()

    def get_food(self, only_new=False):
        if only_new:
            return self.newest_food
        return list(self.locations)