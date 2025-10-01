

import numpy as np

from typing import Dict, List
from itertools import permutations

from snake_sim.map_utils.general import print_map
from snake_sim.environment.types import (
    LoopStartData,
    LoopStepData,
    Coord
)

from snake_sim.loop_observers.loop_data_consumer import LoopDataConsumer


class BFBuilder(LoopDataConsumer):
    """ Receives loop data and can construct framebuffers of the simulation. does not keep all frames in memory, only the current one.
    Creates new frames either next or previous from the current one, depending on the expansion rate. """
    def __init__(self, expansion=1):
        super().__init__()
        self._expansion = expansion
        # step_count and frame_count are to keep track of what frame to return on next_frame/previous_frame calls
        self._backing = False
        self._curr_step_idx = -1
        self._curr_frame_idx = -1
        # non-expanded snake positions in the current step
        self._head_coords: Dict[int, Coord] = {}
        self._tail_coords: Dict[int, Coord] = {}
        self._current_int_frames: List[np.ndarray] = [] # only to keep intermediate frames if expansion > 1

    def notify_start(self, start_data: LoopStartData):
        super().notify_start(start_data)
        init_data = self._start_data.env_init_data
        for s_id, pos in init_data.start_positions.items():
            self._head_coords[s_id] = pos
            self._tail_coords[s_id] = pos
        first_frame = self._expand_frame(init_data.base_map.copy())
        # put the snakes on the map
        for s_id in self._head_coords:
            head = self._ex_coord(self._head_coords[s_id])
            first_frame[head.y, head.x] = init_data.snake_values[s_id]['head_value']
        self._current_int_frames = [first_frame]

    def _ex_coord(self, coord: Coord) -> Coord:
        """ Expand a coordinate according to the expansion factor. """
        return Coord(*coord) * Coord(self._expansion, self._expansion)

    def _get_int_coords(self, start_coord: Coord, direction: Coord) -> Coord:
        """ Get all intermediate coordinates from start_coord to start_coord + direction, according to the expansion factor. """
        new_coords = []
        for i in range(1, self._expansion + 1):
            new_coord = start_coord + direction * Coord(i, i)
            new_coords.append(new_coord)
        return new_coords

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

    def _check_at_end(self, forward: bool) -> bool:
        at_buffer_end = self._check_at_buffers_end(forward)
        if forward:
            return self._curr_step_idx >= len(self._steps) and at_buffer_end
        else:
            return self._curr_step_idx < 0 and at_buffer_end

    def _check_at_buffers_end(self, forward: bool) -> bool:
        if forward:
            return self._curr_frame_idx >= len(self._current_int_frames)
        else:
            return self._curr_frame_idx < 0

    def _get_frame(self, forward: bool) -> np.ndarray:
        idx_direction = 1 if forward else -1
        self._curr_frame_idx += idx_direction
        if self._check_at_end(forward):
            return None
        if self._check_at_buffers_end(forward):
            self._curr_step_idx += idx_direction
            if forward:
                self._current_int_frames = self._create_forward_frames()
                self._curr_frame_idx = 0
            else:
                self._current_int_frames = self._create_backward_frames()
                self._curr_frame_idx = len(self._current_int_frames) - 1
        new_frame = self._current_int_frames[self._curr_frame_idx]
        return new_frame

    def next_frame(self) -> np.ndarray:
        """ Get the next frame in the sequence. Returns None if there are no more frames. """
        if self._curr_step_idx <= 10 and not self._backing:
            print("forwards")
            return self._get_frame(True)
        else:
            self._backing = True
            print("backwards")
            return self._get_frame(False)

    def previous_frame(self) -> np.ndarray:
        """ Get the previous frame in the sequence. Returns None if there are no more frames. """
        return self._get_frame(False)

    def _create_forward_frames(self) -> List[np.ndarray]:
        """ Create frames between current step and next step, according to the expansion factor. """
        step_idx = self._curr_step_idx
        step_data = self._steps[step_idx]
        init_data = self._start_data.env_init_data
        next_frame = self._current_int_frames[-1].copy()
        frames = []
        for food in step_data.food:
            ex_food = self._ex_coord(food)
            next_frame[ex_food.y, ex_food.x] = init_data.food_value
        for i in range(1, self._expansion + 1):
            for s_id in self._head_coords:
                h_dir = step_data.decisions[s_id]
                t_dir = step_data.tail_directions[s_id]
                curr_head = self._ex_coord(self._head_coords[s_id]) + (h_dir * ((i - 1), (i - 1)))
                curr_tail = self._ex_coord(self._tail_coords[s_id]) + (t_dir * ((i - 1), (i - 1)))
                next_head = curr_head + h_dir
                if t_dir != Coord(0, 0): next_frame[*reversed(curr_tail)] = init_data.free_value
                next_frame[*reversed(curr_head)] = init_data.snake_values[s_id]['body_value']
                next_frame[*reversed(next_head)] = init_data.snake_values[s_id]['head_value']
            frames.append(next_frame.copy())
        for s_id in step_data.decisions:
            self._head_coords[s_id] += step_data.decisions[s_id]
            self._tail_coords[s_id] += step_data.tail_directions[s_id]
        return frames

    def _create_backward_frames(self) -> List[np.ndarray]:
        flip_dir = Coord(-1, -1)
        # we need to reset from the next step to the current step (backwards)
        # self._curr_step_idx is currently the step we want to go back to
        next_step_idx = self._curr_step_idx + 1
        next_step = self._steps[next_step_idx]
        current_step = self._steps[self._curr_step_idx] if self._curr_step_idx >= 0 else next_step
        init_data = self._start_data.env_init_data
        # move all snakes back to their previous position
        for s_id in self._head_coords:
            self._head_coords[s_id] -= next_step.decisions[s_id]
            self._tail_coords[s_id] -= next_step.tail_directions[s_id]
        # reset frame from last step
        frames = []
        next_frame = self._current_int_frames[0].copy()
        food_set = set(current_step.food)
        for s_id in self._head_coords:
            s_head = self._head_coords[s_id]
            s_tail = self._tail_coords[s_id]
            ex_head = self._ex_coord(s_head)
            ex_tail = self._ex_coord(s_tail)
            reset_head = ex_head + next_step.decisions[s_id]
            next_frame[reset_head.y, reset_head.x] = init_data.free_value if reset_head not in food_set else init_data.food_value
            next_frame[ex_tail.y, ex_tail.x] = init_data.snake_values[s_id]['body_value']
            next_frame[ex_head.y, ex_head.x] = init_data.snake_values[s_id]['head_value']
        frames.append(next_frame.copy())
        if current_step is next_step:
            return frames
        # Now we need to create the intermediate frames for the current step
        for i in range(1, self._expansion): # since we handled one frame we only need expansion-1 more
            for s_id in self._head_coords:
                h_dir = current_step.decisions[s_id] * flip_dir
                t_dir = current_step.tail_directions[s_id] * flip_dir
                prev_head = self._ex_coord(self._head_coords[s_id]) + (h_dir * (i - 1, i - 1))
                curr_tail = self._ex_coord(self._tail_coords[s_id]) + (t_dir * (i, i))
                curr_head = prev_head + h_dir
                next_frame[*reversed(curr_tail)] = init_data.snake_values[s_id]['body_value']
                next_frame[*reversed(prev_head)] = init_data.free_value
                next_frame[*reversed(curr_head)] = init_data.snake_values[s_id]['head_value']
            frames.append(next_frame.copy())
        frames.reverse() # we created them backwards, so reverse the list
        return frames


    def notify_step(self, step_data: LoopStepData):
        super().notify_step(step_data)
        frame = self.next_frame()
        print_map(
            frame,
            self._start_data.env_init_data.free_value,
            self._start_data.env_init_data.food_value,
            self._start_data.env_init_data.blocked_value,
            255,
            255,
        )
