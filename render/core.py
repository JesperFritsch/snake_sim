import json
import math
import numpy as np
from pathlib import Path
from collections import deque

from utils import coord_op, coord_cmp

class SnakeRepresentation:
    def __init__(self, snake_id, head_color, body_color, expand_factor):
        self.snake_id = snake_id
        self.head_color = head_color
        self.body_color = body_color
        self.expand_factor = expand_factor
        self.last_head = None
        self.last_tail = None
        self.prev_expand_head = None
        self.body = deque()

    def update(self, step_snake_data, expand_step):
        head_dir = step_snake_data['head_dir']
        if step_snake_data['curr_head'] == self.last_head:
            return
        self.prev_expand_head = coord_op(step_snake_data['prev_head'], (self.expand_factor, self.expand_factor), '*')
        if len(self.body) == 0:
            self.body.appendleft(self.prev_expand_head)
        dir_mult = coord_op(head_dir, (expand_step, expand_step), '*')
        next_head = coord_op(self.prev_expand_head, dir_mult, '+')
        self.body.appendleft(next_head)
        if tuple(step_snake_data['tail_dir']) != (0, 0):
            self.last_tail = self.body.pop()
        else:
            self.last_tail = None
        if expand_step == self.expand_factor:
            self.last_head = step_snake_data['curr_head']


class FrameBuilder:
    def __init__(self, run_meta_data, expand_factor=2, offset=(0, 0)):
        self.width = run_meta_data['width']
        self.height = run_meta_data['height']
        self.base_map = run_meta_data['base_map']
        self.offset = offset
        self.offset_x, self.offset_y = self.offset
        self.free_value = run_meta_data['free_value']
        self.food_value = run_meta_data['food_value']
        self.blocked_value = run_meta_data['blocked_value']
        self.color_mapping = run_meta_data['color_mapping']
        self.expand_factor = expand_factor
        self.last_food = set()
        self.frameshape = ((self.height * self.expand_factor) + self.offset_y, (self.width * self.expand_factor) + self.offset_x, 3)
        self.snake_reps = {}
        for snake_data in run_meta_data['snake_data']:
            snake_id = snake_data['snake_id']
            head_color = tuple(snake_data['head_color'])
            body_color = tuple(snake_data['body_color'])
            self.snake_reps[snake_id] = SnakeRepresentation(snake_id, head_color, body_color, expand_factor)
        self.set_base_frame()

    def set_base_frame(self):
        neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        self.last_frame = np.full(self.frameshape, self.color_mapping[self.free_value], dtype=np.uint8)
        for y in range(self.height):
            for x in range(self.width):
                if self.base_map[y * self.width + x] == self.blocked_value:
                    ex_x, ex_y = coord_op((x, y), (self.expand_factor, self.expand_factor), '*')
                    ex_x, ex_y = coord_op((ex_x, ex_y), self.offset, '+')
                    self.last_frame[ex_y, ex_x] = self.color_mapping[self.blocked_value]
                    if self.expand_factor > 1:
                        # if the expand factor is greater than 1, we need to color the neighbors of the blocked cell in the frame
                        for n in neighbors:
                            nx, ny = coord_op((x, y), n, '+')
                            if 0 <= nx < self.width and 0 <= ny < self.height and self.base_map[ny * self.width + nx] == self.blocked_value:
                                ex_nx, ex_ny = coord_op((ex_x, ex_y), n, '+')
                                self.last_frame[ex_ny, ex_nx] = self.color_mapping[self.blocked_value]


    def step_to_pixel_changes(self, step_data):
        changes = []
        current_food = set([tuple(x) for x in step_data['food']])
        new_food = current_food - self.last_food
        gone_food = self.last_food - current_food
        sub_changes = []
        for food in new_food:
            x, y = coord_op(food, (self.expand_factor, self.expand_factor), '*')
            sub_changes.append(((x, y), self.color_mapping[self.food_value]))
        for food in gone_food:
            if not any([coord_cmp(food, snake_rep.last_head) for snake_rep in self.snake_reps.values()]):
                x, y = coord_op(food, (self.expand_factor, self.expand_factor), '*')
                sub_changes.append(((x, y), self.color_mapping[self.free_value]))
        self.last_food = current_food
        for s in range(1, self.expand_factor + 1):
            for snake_data in step_data['snakes']:
                snake_id = snake_data['snake_id']
                self.snake_reps[snake_id].update(snake_data, s)
            for snake_id in self.snake_reps:
                snake_rep = self.snake_reps[snake_id]
                last_tail = snake_rep.last_tail
                head = tuple(snake_rep.body[0])
                if last_tail is not None:
                    sub_changes.append((tuple(last_tail), self.color_mapping[self.free_value]))
                sub_changes.append((tuple(snake_rep.body[1]), snake_rep.body_color))
                sub_changes.append((head, snake_rep.head_color))
            sub_changes = [(coord_op(coord, self.offset, '+'), color) for coord, color in sub_changes]
            changes.append(list(set(sub_changes)))
            sub_changes = []
        return changes


    def step_to_frames(self, step_data):
        frames = []
        for changes in self.step_to_pixel_changes(step_data):
            frame = self.last_frame.copy()
            for (x, y), color in changes:
                frame[y, x] = color
            self.last_frame = frame
            frames.append(frame)
        return frames

    def frames_from_rundata(self, rundata):
        frames = []
        for body_coords in rundata:
            frames.append(put_snake_in_frame(self.last_frame.copy(), body_coords, (255, 0, 0), self.expand_factor, self.offset))
        return frames

def pixel_changes_from_runfile(filepath, expand_factor=2, offset=(1, 1)):
    pixel_changes = []
    with open(Path(filepath)) as run_file:
        run_data = json.load(run_file)
        metadata = run_data.copy()
        metadata['color_mapping'] = {int(k): tuple(v) for k, v in metadata['color_mapping'].items()}
        del metadata['steps']
        frame_builder = FrameBuilder(metadata, expand_factor, offset)
        for step_nr, step_data in run_data['steps'].items():
            pixel_changes.extend(frame_builder.step_to_pixel_changes(step_data))
    return pixel_changes


def put_snake_in_frame(frame, snake_coords, color, expand_factor=2, offset=(0, 0)):
    for i, coord in enumerate(snake_coords):
        s_dirs = []
        if i > 0:
            s_dirs.append(coord_op(snake_coords[i-1], coord, '-'))
        elif i < len(snake_coords) - 1:
            s_dirs.append(coord_op(snake_coords[i+1], coord, '-'))
        expanded = coord_op(coord, (expand_factor, expand_factor), '*')
        x, y = coord_op(expanded, offset, '+')
        frame[y, x] = color
        for s_dir in s_dirs:
            for _ in range(math.ceil(expand_factor / 2) + 1):
                con_x, con_y = coord_op((x, y), s_dir, '+')
                frame[con_y, con_x] = color
    return frame
