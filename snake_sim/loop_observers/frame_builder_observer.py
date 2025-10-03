

import numpy as np

from typing import Dict, List
from itertools import permutations

from snake_sim.environment.types import (
    LoopStartData,
    Coord
)

from snake_sim.loop_observers.consumer_observer import ConsumerObserver


class NoMoreSteps(Exception):
    pass


class CurrentIsFirst(Exception):
    pass


class FrameBuilderObserver(ConsumerObserver):
    """ Receives loop data and can construct framebuffers of the simulation. does not keep all frames in memory, only the current one.
    Creates new frames either next or previous from the current one. """
    def __init__(self, expansion=1):
        super().__init__()
        self._expansion = expansion
        # step_count and frame_count are to keep track of what frame to return on next_frame/previous_frame calls
        # expanded snake positions in the current step
        self._head_coords: Dict[int, Coord] = {}
        self._tail_coords: Dict[int, Coord] = {}
        self._current_frame: np.ndarray = None
        self._current_frame_idx = 0
        self._curr_step = 0
        self._curr_step_idx = 0
        self._backing = False

    def notify_start(self, start_data: LoopStartData):
        super().notify_start(start_data)
        init_data = self._start_data.env_init_data
        self._current_frame = self._expand_frame(init_data.base_map.copy())
        for s_id, pos in init_data.start_positions.items():
            s_pos = self._ex_coord(pos)
            self._head_coords[s_id] = s_pos
            self._tail_coords[s_id] = s_pos
            self._current_frame[s_pos.y, s_pos.x] = init_data.snake_values[s_id]['head_value']

    def _ex_coord(self, coord: Coord) -> Coord:
        """ Expand a coordinate according to the expansion factor. """
        return Coord(*coord) * Coord(self._expansion, self._expansion)

    def _expand_frame(self, frame: np.ndarray):
        height, width = frame.shape
        free_value = self._start_data.env_init_data.free_value
        blocked_value = self._start_data.env_init_data.blocked_value
        neighbors = [Coord(*c) for c in permutations([-1, 0, 1], 2) if abs(sum(c)) == 1]
        expanded_frame = np.full(
            (frame.shape[0]*self._expansion, frame.shape[1]*self._expansion),
            free_value,
            dtype=frame.dtype
        )
        for y in range(height):
            for x in range(width):
                coord = Coord(x, y)
                if frame[y, x] == blocked_value:
                    ex_coord = self._ex_coord(coord)
                    expanded_frame[ex_coord.y, ex_coord.x] = blocked_value
                    if self._expansion > 1:
                        # if the expand factor is greater than 1, we need to color the neighbors of the blocked cell in the frame
                        for n in neighbors:
                            n_coord = coord + n
                            if 0 <= n_coord.x < width and 0 <= n_coord.y < height and frame[n_coord.y, n_coord.x] == blocked_value:
                                ex_n_coord = ex_coord + n
                                expanded_frame[ex_n_coord.y, ex_n_coord.x] = blocked_value
        return expanded_frame

    def get_current_frame(self):
        return self._current_frame.copy()

    def get_next_frame(self):
        self._goto_next_frame()
        return self._current_frame.copy()

    def get_prev_frame(self):
        self._goto_prev_frame()
        return self._current_frame.copy()

    def get_frame(self, frame_idx: int):
        self._goto_frame(frame_idx)
        return self._current_frame.copy()

    def get_frame_for_step(self, step_idx: int):
        frame_idx = step_idx * self._expansion
        return self.get_frame(frame_idx)

    def _goto_frame(self, frame_idx: int):
        idx_delta = frame_idx - self._current_frame_idx
        while idx_delta != 0:
            if idx_delta > 0:
                self._goto_next_frame()
                idx_delta -= 1
            else:
                self._goto_prev_frame()
                idx_delta += 1

    def _goto_next_frame(self) -> List[np.ndarray]:
        """ Create frames between current step and next step, according to the expansion factor. """
        if self._current_frame_idx // self._expansion >= len(self._steps):
            if self._stop_data is not None:
                raise StopIteration("No more frames available")
            raise NoMoreSteps("Need to receive more steps to generate frames")
        self._curr_step_idx = self._current_frame_idx // self._expansion
        self._curr_step = self._steps[self._curr_step_idx]
        self._current_frame_idx += 1
        step_data = self._curr_step
        init_data = self._start_data.env_init_data
        for food in step_data.new_food:
            ex_food = self._ex_coord(food)
            self._current_frame[ex_food.y, ex_food.x] = init_data.food_value
        for s_id in step_data.decisions:
            curr_head = self._head_coords[s_id]
            curr_tail = self._tail_coords[s_id]
            self._head_coords[s_id] += step_data.decisions[s_id]
            self._tail_coords[s_id] += step_data.tail_directions[s_id]
            new_tail = self._tail_coords[s_id]
            new_head = self._head_coords[s_id]
            if curr_tail.manhattan_distance(new_tail) != 0:
                self._current_frame[*reversed(curr_tail)] = init_data.free_value
            self._current_frame[*reversed(curr_head)] = init_data.snake_values[s_id]['body_value']
            self._current_frame[*reversed(new_head)] = init_data.snake_values[s_id]['head_value']

    def _goto_prev_frame(self) -> List[np.ndarray]:
        if self._current_frame_idx <= 0:
            self._current_frame_idx = 0
            raise CurrentIsFirst("Current frame is the first frame")
        self._current_frame_idx -= 1
        self._curr_step_idx = self._current_frame_idx // self._expansion
        self._curr_step = self._steps[self._curr_step_idx]
        step_data = self._curr_step
        init_data = self._start_data.env_init_data
        food_set = set([self._ex_coord(f) for f in step_data.new_food])
        for s_id in step_data.decisions:
            curr_head = self._head_coords[s_id]
            self._head_coords[s_id] -= step_data.decisions[s_id]
            self._tail_coords[s_id] -= step_data.tail_directions[s_id]
            new_head = self._head_coords[s_id]
            new_tail = self._tail_coords[s_id]
            self._current_frame[curr_head.y, curr_head.x] = init_data.free_value if curr_head not in food_set else init_data.food_value
            self._current_frame[new_tail.y, new_tail.x] = init_data.snake_values[s_id]['body_value']
            self._current_frame[new_head.y, new_head.x] = init_data.snake_values[s_id]['head_value']