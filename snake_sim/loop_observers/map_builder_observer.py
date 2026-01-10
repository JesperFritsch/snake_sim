

import numpy as np

from typing import Dict, List
from itertools import permutations

from snake_sim.environment.types import (
    LoopStartData,
    LoopStepData,
    Coord
)

from snake_sim.loop_observers.consumer_observer import ConsumerObserver


class NoMoreSteps(Exception):
    pass


class CurrentIsFirst(Exception):
    pass


class MapBuilderObserver(ConsumerObserver):
    """ Receives loop data and can construct map buffers of the simulation. does not keep all maps in memory, only the current one.
    Creates new map, either next or previous from the current one. """
    def __init__(self, expansion=1):
        super().__init__()
        self._expansion = expansion
        # step_count and map_count are to keep track of what map to return on next_map/previous_map calls
        # expanded snake positions in the current step
        self._ex_head_coords: Dict[int, Coord] = {}
        self._ex_tail_coords: Dict[int, Coord] = {}
        self._head_coords: Dict[int, Coord] = {}
        self._current_map: np.ndarray = None
        self._current_map_idx = 0
        self._backing = False

    def notify_start(self, start_data: LoopStartData):
        super().notify_start(start_data)
        init_data = self._start_data.env_meta_data
        self._current_map = self._expand_map(init_data.base_map.copy())
        for s_id, pos in init_data.start_positions.items():
            s_pos = self._ex_coord(pos)
            self._ex_head_coords[s_id] = s_pos
            self._ex_tail_coords[s_id] = s_pos
            self._head_coords[s_id] = pos
            self._current_map[s_pos.y, s_pos.x] = init_data.snake_values[s_id]['head_value']

    def reset(self):
        self._current_map = None
        self._ex_head_coords.clear()
        self._ex_tail_coords.clear()
        self._head_coords.clear()
        self._current_map_idx = 0
        self._backing = False
        return super().reset()

    def _ex_coord(self, coord: Coord) -> Coord:
        """ Expand a coordinate according to the expansion factor. """
        return Coord(*coord) * Coord(self._expansion, self._expansion)

    def _expand_map(self, map: np.ndarray):
        height, width = map.shape
        free_value = self._start_data.env_meta_data.free_value
        blocked_value = self._start_data.env_meta_data.blocked_value
        neighbors = [Coord(*c) for c in permutations([-1, 0, 1], 2) if abs(sum(c)) == 1]
        expanded_map = np.full(
            (map.shape[0]*self._expansion, map.shape[1]*self._expansion),
            free_value,
            dtype=map.dtype
        )
        for y in range(height):
            for x in range(width):
                coord = Coord(x, y)
                if map[y, x] == blocked_value:
                    ex_coord = self._ex_coord(coord)
                    expanded_map[ex_coord.y, ex_coord.x] = blocked_value
                    if self._expansion > 1:
                        # if the expand factor is greater than 1, we need to color the neighbors of the blocked cell in the map
                        for n in neighbors:
                            n_coord = coord + n
                            if 0 <= n_coord.x < width and 0 <= n_coord.y < height and map[n_coord.y, n_coord.x] == blocked_value:
                                ex_n_coord = ex_coord + n
                                expanded_map[ex_n_coord.y, ex_n_coord.x] = blocked_value
        return expanded_map

    def get_current_map(self) -> np.ndarray:
        return self._current_map.copy()

    def get_next_map(self) -> np.ndarray:
        self._goto_next_map()
        return self.get_current_map()

    def get_prev_map(self) -> np.ndarray:
        self._goto_prev_map()
        return self.get_current_map()

    def get_map(self, map_idx: int) -> np.ndarray:
        self._goto_map(map_idx)
        return self.get_current_map()

    def get_map_for_step(self, step_idx: int) -> np.ndarray:
        map_idx = step_idx * self._expansion
        return self.get_map(map_idx)

    def get_max_map_idx(self):
        return len(self._steps) * self._expansion
    
    def get_max_step_idx(self):
        return len(self._steps)

    def get_current_map_idx(self):
        return self._current_map_idx

    def get_current_step_idx(self) -> int:
        return self._current_map_idx // self._expansion

    def _goto_map(self, map_idx: int):
        idx_delta = map_idx - self._current_map_idx
        while idx_delta != 0:
            if idx_delta > 0:
                self._goto_next_map()
                idx_delta -= 1
            else:
                self._goto_prev_map()
                idx_delta += 1

    def _goto_next_map(self):
        """ Create maps between current step and next step, according to the expansion factor. """
        curr_step_idx = self.get_current_step_idx()
        if curr_step_idx >= len(self._steps):
            if self._stop_data is not None:
                raise StopIteration("No more maps available")
            raise NoMoreSteps("Need to receive more steps to generate maps")
        step_data = self._steps[curr_step_idx]
        self._current_map_idx += 1
        init_data = self._start_data.env_meta_data
        if ((self._current_map_idx - 1) % self._expansion) == 0:
            for s_id in step_data.decisions:
                self._head_coords[s_id] += step_data.decisions[s_id]

            for food in step_data.new_food:
                ex_food = self._ex_coord(food)
                self._current_map[ex_food.y, ex_food.x] = init_data.food_value

            for food in set(step_data.removed_food) - set(self._head_coords.values()):
                ex_food = self._ex_coord(food)
                self._current_map[ex_food.y, ex_food.x] = init_data.free_value

        for s_id in step_data.decisions:
            curr_head = self._ex_head_coords[s_id]
            curr_tail = self._ex_tail_coords[s_id]
            decision = step_data.decisions[s_id]
            self._ex_head_coords[s_id] += decision
            self._ex_tail_coords[s_id] += step_data.tail_directions[s_id]
            new_tail = self._ex_tail_coords[s_id]
            new_head = self._ex_head_coords[s_id]
            if curr_tail.manhattan_distance(new_tail) != 0:
                self._current_map[*reversed(curr_tail)] = init_data.free_value
            if step_data.lengths[s_id] > 1:
                self._current_map[*reversed(curr_head)] = init_data.snake_values[s_id]['body_value']
            self._current_map[*reversed(new_head)] = init_data.snake_values[s_id]['head_value']


    def _goto_prev_map(self):
        if self._current_map_idx <= 0:
            self._current_map_idx = 0
            raise CurrentIsFirst("Current map is the first map")
        self._current_map_idx -= 1
        curr_step_idx = self.get_current_step_idx()
        step_data = self._steps[curr_step_idx]
        init_data = self._start_data.env_meta_data

        for s_id in step_data.decisions:
            curr_head = self._ex_head_coords[s_id]
            self._ex_head_coords[s_id] -= step_data.decisions[s_id]
            self._ex_tail_coords[s_id] -= step_data.tail_directions[s_id]
            new_head = self._ex_head_coords[s_id]
            new_tail = self._ex_tail_coords[s_id]
            self._current_map[curr_head.y, curr_head.x] = init_data.free_value# if curr_head not in food_set else init_data.food_value
            self._current_map[new_tail.y, new_tail.x] = init_data.snake_values[s_id]['body_value']
            self._current_map[new_head.y, new_head.x] = init_data.snake_values[s_id]['head_value']

        if (self._current_map_idx % self._expansion) == 0:
            for s_id in step_data.decisions:
                self._head_coords[s_id] -= step_data.decisions[s_id]

            for food in step_data.new_food:
                ex_food = self._ex_coord(food)
                self._current_map[ex_food.y, ex_food.x] = init_data.free_value

            for food in step_data.removed_food:
                ex_food = self._ex_coord(food)
                self._current_map[ex_food.y, ex_food.x] = init_data.food_value
