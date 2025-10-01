

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
        self._curr_step_idx = 0
        self._curr_frame_idx = 0
        # non-expanded snake positions in the current step
        self._head_coords: Dict[int, Coord] = {}
        self._tails_coords: Dict[int, Coord] = {}
        # directions of movement to get to where they are in the current step
        self._head_directions: Dict[int, Coord] = {}
        self._tail_directions: Dict[int, Coord] = {}
        self._current_int_frames: List[np.ndarray] = [] # only to keep intermediate frames if expansion > 1

    def notify_start(self, start_data: LoopStartData):
        super().notify_start(start_data)
        init_data = self._start_data.env_init_data
        for s_id, pos in init_data.start_positions.items():
            self._head_coords[s_id] = pos
            self._tails_coords[s_id] = pos
            self._head_directions[s_id] = Coord(0, 0)
            self._tail_directions[s_id] = Coord(0, 0)
        first_frame = self._expand_frame(init_data.base_map.copy())
        # put the snakes on the map
        for s_id in self._head_coords:
            head = self._ex_coord(self._head_coords[s_id])
            first_frame[head.y, head.x] = init_data.snake_values[s_id]['head_value']
        self._current_int_frames = [first_frame]
    
    # def notify_step(self, step_data: LoopStepData):
    #     super().notify_step(step_data)
    
    def _ex_coord(self, coord: Coord) -> Coord:
        """ Expand a coordinate according to the expansion factor. """
        return coord * Coord(self._expansion, self._expansion)

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

    def _check_at_end(self, idx_direction: int) -> bool:
        at_buffer_end = self._check_at_buffers_end(idx_direction)
        if idx_direction == 1:
            return self._curr_step_idx >= len(self._steps) and at_buffer_end
        else:
            return self._curr_step_idx <= 0 and at_buffer_end

    def _check_at_buffers_end(self, idx_direction: int) -> bool:
        if idx_direction == 1:
            return self._curr_frame_idx >= len(self._current_int_frames)
        else:
            return self._curr_frame_idx <= 0

    def _get_frame(self, idx_direction: int) -> np.ndarray:
        if self._check_at_end(idx_direction):
            return None
        if self._check_at_buffers_end(idx_direction):
            self._current_int_frames = self._create_int_frames(idx_direction)
            self._curr_step_idx += idx_direction
            self._curr_frame_idx = 0 if idx_direction == 1 else len(self._current_int_frames) - 1
        new_frame = self._current_int_frames[self._curr_frame_idx]
        self._curr_frame_idx += idx_direction
        return new_frame
    
    def next_frame(self) -> np.ndarray:
        """ Get the next frame in the sequence. Returns None if there are no more frames. """
        return self._get_frame(1)
    
    def previous_frame(self) -> np.ndarray:
        """ Get the previous frame in the sequence. Returns None if there are no more frames. """
        return self._get_frame(-1)

    def _create_int_frames(self, idx_direction: int) -> List[np.ndarray]:
        """ Create frames between current step and next step, according to the expansion factor. """
        direction_flip = Coord(idx_direction, idx_direction) # to get the direction of movement right when going backwards
        step_idx = self._curr_step_idx + idx_direction
        step_data = self._steps[step_idx]
        init_data = self._start_data.env_init_data
        current_frame = self._current_int_frames[-1] if idx_direction == 1 else self._current_int_frames[0]
        next_frame = current_frame.copy()
        frames = []
        intermediate_heads = {s_id: self._ex_coord(self._head_coords[s_id]) for s_id in self._head_coords}
        intermediate_tails = {s_id: self._ex_coord(self._tails_coords[s_id]) for s_id in self._tails_coords}
        food_set = set(step_data.food)
        for _ in range(self._expansion):
            for s_id in self._head_coords:
                int_head = intermediate_heads[s_id]
                int_tail = intermediate_tails[s_id]
                head_direction = self._head_directions[s_id] * direction_flip
                tail_direction = self._tail_directions[s_id] * direction_flip
                head_replace_value = init_data.snake_values[s_id]['body_value'] if idx_direction == 1 else init_data.free_value
                tail_replace_value = init_data.free_value if idx_direction == 1 else init_data.snake_values[s_id]['body_value']
                new_head_coord = int_head + head_direction
                new_tail_coord = int_tail + tail_direction
                next_frame[*reversed(new_head_coord)] = init_data.snake_values[s_id]['head_value']
                next_frame[*reversed(new_tail_coord)] = init_data.snake_values[s_id]['body_value']
                next_frame[*reversed(int_head)] = head_replace_value if head_replace_value not in food_set else init_data.food_value
                next_frame[*reversed(int_tail)] = tail_replace_value
                intermediate_heads[s_id] = new_head_coord
                intermediate_tails[s_id] = new_tail_coord
            frame_copy = next_frame.copy()
            if idx_direction == 1:
                frames.append(frame_copy)
            else:
                frames.insert(0, frame_copy)
        for s_id in step_data.decisions:
            self._head_directions[s_id] = step_data.decisions[s_id]
            self._tail_directions[s_id] = step_data.tail_directions[s_id]
            self._head_coords[s_id] = self._head_coords[s_id] + self._head_directions[s_id]
            self._tails_coords[s_id] = self._tails_coords[s_id] + self._tail_directions[s_id]
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
