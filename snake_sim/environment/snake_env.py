import json
import numpy as np
import logging
from typing import Optional, Dict, Deque, Tuple, Union
from PIL import Image
from pathlib import Path
from importlib import resources
from collections import deque

from snake_sim.environment.food_handlers import IFoodHandler
from snake_sim.environment.interfaces.snake_env_interface import ISnakeEnv
from snake_sim.environment.types import EnvData, EnvInitData, DotDict, Coord

from snake_sim.map_utils.general import print_map, convert_png_to_map

log = logging.getLogger(Path(__file__).stem)

with resources.open_text('snake_sim.config', 'default_config.json') as config_file:
    config = DotDict(json.load(config_file))


class SnakeRep:
    def __init__(self, id: int, h_value: int, b_value: int, start_position: Coord, start_length: int=1):
        self.move_count = 0
        self.last_ate = 0
        self.id = id
        self.head_value = h_value
        self.body_value = b_value
        self._length = start_length
        self.body: Deque[Coord] = deque([start_position])
        self.is_alive = True

    def kill(self):
        self.is_alive = False

    def grow(self):
        self._length += 1

    def eat(self):
        self.last_ate = self.move_count
        self.grow()

    def move(self, direction: Coord):
        """ Moves the snake in the given direction. returns True if the snake grew, otherwise False """
        prev_len = len(self.body)
        self._move(direction)
        return prev_len < len(self.body)

    def _move(self, direction: Coord):
        self.move_count += 1
        self.body.appendleft(self.get_head() + direction)
        while len(self.body) > self._length:
            self.body.pop()

    def get_head(self) -> Coord:
        return self.body[0]

    def get_tail(self) -> Coord:
        return self.body[-1]


class SnakeEnv(ISnakeEnv):
    def __init__(self, width, height, free_value, blocked_value, food_value):
        self._width = width
        self._height = height
        self._free_value = free_value
        self._blocked_value = blocked_value
        self._food_value = food_value
        self._map = np.full((height, width), self._free_value, dtype=np.uint8)
        self._base_map = np.copy(self._map)
        self._snake_reps: Dict[int, SnakeRep] = {}
        self._used_map_values = set([self._free_value, self._blocked_value, self._food_value])

    def add_snake(self, id: int, start_position: Optional[Coord]=None, start_length: int=1) -> Coord:
        if start_position is None:
            start_position = self._random_free_tile()
        head_value, body_value = self._assign_map_values()
        snake_rep = SnakeRep(id, head_value, body_value, start_position, start_length)
        self._snake_reps[id] = snake_rep
        self._place_snake_on_map(snake_rep)
        return start_position

    def _assign_map_values(self) -> Tuple[int, int]: # returns head_value, body_value
        head_value = self._find_unused_map_value()
        self._used_map_values.add(head_value)
        body_value = self._find_unused_map_value()
        self._used_map_values.add(body_value)
        return head_value, body_value

    def _find_unused_map_value(self) -> int:
        for i in range(0xff):
            if i not in self._used_map_values:
                return i
        raise ValueError("No more map values available")

    def _random_free_tile(self) -> Coord:
        while True:
            rand_coord = Coord(np.random.randint(1, self._width - 1), np.random.randint(1, self._height - 1))
            x, y = rand_coord
            if self._map[y, x] == self._free_value:
                return rand_coord

    def _place_snake_on_map(self, snake_rep: SnakeRep):
        for i, (x, y) in enumerate(snake_rep.body):
            self._map[y, x] = snake_rep.id + (0 if i == 0 else 1)

    def _remove_snake_from_map(self, snake_rep: SnakeRep):
        for x, y in snake_rep.body:
            self._map[y, x] = self._free_value

    def move_snake(self, id: int, direction: Union[Coord, None]) -> Tuple[bool, bool, Coord]:
        # direction is expected to be a Coord like (1, 0) for right
        # returns (alive, grew)
        snake_rep = self._snake_reps[id]
        if self._is_valid_move(id, direction):
            current_head = snake_rep.get_head()
            next_tile = current_head + direction
            if self._is_food_tile(next_tile):
                self._food_handler.remove(next_tile, self._map)
                snake_rep.eat()
            old_tail = snake_rep.get_tail()
            grow = snake_rep.move(direction)
            self._update_snake_on_map(id, old_tail)
            tail_direction = snake_rep.get_tail() - old_tail
            return True, grow, tail_direction
        else:
            snake_rep.kill()
            return False, False, Coord(0, 0)

    def _update_snake_on_map(self, id, old_tail):
        snake_rep = self._snake_reps[id]
        new_head = snake_rep.get_head()
        old_head = snake_rep.body[1]
        new_tail = snake_rep.get_tail()
        self._map[new_head[1], new_head[0]] = snake_rep.head_value
        self._map[old_head[1], old_head[0]] = snake_rep.body_value
        self._map[new_tail[1], new_tail[0]] = snake_rep.body_value
        if old_tail != new_tail:
            self._map[old_tail[1], old_tail[0]] = self._free_value

    def _is_inside(self, coord: Coord):
        x, y = coord
        return 0 <= x < self._width and 0 <= y < self._height

    def _is_free_tile(self, coord: Coord):
        x, y = coord
        return self._is_inside(coord) and self._map[y, x] <= self._free_value

    def _is_food_tile(self, coord: Coord):
        x, y = coord
        return self._is_inside(coord) and self._map[y, x] == self._food_value

    def _is_valid_move(self, id: int, direction: Coord):
        if direction is None:
            return False
        next_tile = self._snake_reps[id].get_head() + direction
        return self._is_free_tile(next_tile) and direction in config.DIRS.values()

    def get_map(self):
        return np.copy(self._map)

    def get_base_map(self):
        return np.copy(self._base_map)

    def get_food(self):
        return self._food_handler.get_food()

    def get_head_positions(self, only_alive: bool=True) -> Dict[int, Coord]:
        return {id: snake_rep.get_head() for id, snake_rep in self._snake_reps.items() if (not only_alive or snake_rep.is_alive)}

    def get_env_data(self, for_id: Optional[int] = None) -> EnvData:
        # id is not used yet, but it is preparing for being able to send different data to different snakes
        return EnvData(
            self.get_map().tobytes(),
            {id: {'is_alive': snake_rep.is_alive, 'length': len(snake_rep.body)} for id, snake_rep in self._snake_reps.items()},
            self.get_food()
        )

    def get_init_data(self) -> EnvInitData:
        return EnvInitData(
            self._height,
            self._width,
            self._free_value,
            self._blocked_value,
            self._food_value,
            {s_rep.id: {"head_value": s_rep.head_value, "body_value": s_rep.body_value} for s_rep in self._snake_reps.values()},
            {s_rep.id: s_rep.get_head() for s_rep in self._snake_reps.values()},
            self.get_base_map())

    def resize(self, height, width):
        self._height = height
        self._width = width
        self._map.resize((height, width))
        self._base_map.resize((height, width))
        self._food_handler.resize(height, width)

    def load_map(self, map_img_path: str):
        base_map = convert_png_to_map(
            map_img_path,
            {
                (0,0,0,0): self._free_value,
                (255,0,0,255): self._food_value,
                (0,0,0,255): self._blocked_value
            },
        )
        self.resize(*base_map.shape)
        self._base_map[:] = base_map
        self._map = self.get_base_map()

    def set_food_handler(self, food_handler: IFoodHandler):
        if not isinstance(food_handler, IFoodHandler):
            raise ValueError("food_handler must be an instance of IFoodHandler")
        self._food_handler = food_handler
        self._food_handler.resize(self._width, self._height)

    def update_food(self):
        self._food_handler.update(self._map)

    def steps_since_any_ate(self):
        if any(snake_rep.is_alive for snake_rep in self._snake_reps.values()):
            return min(snake_rep.move_count - snake_rep.last_ate for snake_rep in self._snake_reps.values() if snake_rep.is_alive)
        else:
            return 0

    def print_map(self):
        print_map(
            self._map,
            self._free_value,
            self._food_value,
            self._blocked_value,
            300,
            300
        )
