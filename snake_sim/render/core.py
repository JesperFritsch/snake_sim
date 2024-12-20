import json
import math
import numpy as np
from typing import Dict
from pathlib import Path
from collections import deque

from ..utils import coord_op, coord_cmp


class OutOfSyncError(Exception):
    pass


class SnakeRepresentation:
    def __init__(self, snake_id, expand_factor):
        self.snake_id = snake_id
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
        if step_snake_data['did_grow']:
            self.last_tail = None
        else:
            self.last_tail = self.body.pop()
        if expand_step == self.expand_factor:
            self.last_head = step_snake_data['curr_head']

    def set_full_body(self, body_coords):
        self.body = deque()
        for i, curr_coord in enumerate(reversed(body_coords)[1:]):
            prev_coord = body_coords[i - 1]
            snake_data = {}
            snake_data['prev_head'] = prev_coord
            snake_data['curr_head'] = curr_coord
            snake_data['head_dir'] = coord_op(curr_coord, prev_coord, '-')
            for n in range(1, self.expand_factor + 1):
                self.update(snake_data, n)


class FrameBuilder:
    def __init__(self, run_meta_data, expand_factor=2, offset=(1, 1)):
        self.width = run_meta_data['width']
        self.height = run_meta_data['height']
        self.base_map = np.array(run_meta_data['base_map'])
        self.offset = offset
        self.offset_x, self.offset_y = self.offset
        self.free_value = run_meta_data['free_value']
        self.food_value = run_meta_data['food_value']
        self.blocked_value = run_meta_data['blocked_value']
        self.color_mapping = {int(k): tuple(v) for k, v in run_meta_data['color_mapping'].items()}
        print(self.color_mapping)
        print(run_meta_data['snake_values'])
        self.snake_values = {int(k): {"body_value": v["body_value"], "head_value": v["head_value"]} for k, v in run_meta_data['snake_values'].items()}
        print(self.snake_values)
        self.expand_factor = expand_factor
        self.last_food = set()
        self.last_handled_step = -1
        self.frameshape = ((self.height * self.expand_factor) + self.offset_y, (self.width * self.expand_factor) + self.offset_x, 3)
        self.snake_reps: Dict[int, SnakeRepresentation] = {}
        for snake_id in run_meta_data['snakes']:
            snake_id = int(snake_id)
            self.snake_reps[snake_id] = SnakeRepresentation(snake_id, expand_factor)
        self.set_base_frame()

    def set_base_frame(self):
        neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        self.last_frame = np.full(self.frameshape, self.color_mapping[self.free_value], dtype=np.uint8)
        for y in range(self.height):
            for x in range(self.width):
                if self.base_map[y, x] == self.blocked_value:
                    ex_x, ex_y = coord_op((x, y), (self.expand_factor, self.expand_factor), '*')
                    ex_x, ex_y = coord_op((ex_x, ex_y), self.offset, '+')
                    self.last_frame[ex_y, ex_x] = self.color_mapping[self.blocked_value]
                    if self.expand_factor > 1:
                        # if the expand factor is greater than 1, we need to color the neighbors of the blocked cell in the frame
                        for n in neighbors:
                            nx, ny = coord_op((x, y), n, '+')
                            if 0 <= nx < self.width and 0 <= ny < self.height and self.base_map[ny, nx] == self.blocked_value:
                                ex_nx, ex_ny = coord_op((ex_x, ex_y), n, '+')
                                self.last_frame[ex_ny, ex_nx] = self.color_mapping[self.blocked_value]


    def step_to_pixel_changes(self, step_data):
        if step_data['step'] != self.last_handled_step + 1:
            raise OutOfSyncError(f"Step nr '{step_data['step']}' is not the next step in the sequence. Last handled step was '{self.last_handled_step}'")
        self.last_handled_step = step_data['step']
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
                sub_changes.append((tuple(snake_rep.body[1]), self.color_mapping[self.snake_values[snake_id]["body_value"]]))
                sub_changes.append((head, self.color_mapping[self.snake_values[snake_id]["head_value"]]))
            sub_changes = [(coord_op(coord, self.offset, '+'), color) for coord, color in sub_changes]
            changes.append(list(set(sub_changes)))
            sub_changes = []
        return changes

    def full_step_to_pixel_data(self, step_data):
        self.last_handled_step = step_data['step']
        self.set_base_frame()
        snake_data = step_data['snakes']
        food_data = step_data['food']
        food_color = self.color_mapping[self.food_value]
        coord_colors = {}
        for food in food_data:
            expanded_c = coord_op(food, (self.expand_factor, self.expand_factor), '*')
            x, y = coord_op(expanded_c, self.offset, '+')
            self.last_frame[y, x] = food_color
            coord_colors[(x, y)] = food_color
        for snake in snake_data:
            snake_coords = snake['body']
            snake_rep = self.snake_reps[snake['snake_id']]
            snake_rep.set_full_body(snake_coords)
            body_color = self.color_mapping[self.snake_values[snake_rep.snake_id]["body_value"]]
            head_color = self.color_mapping[self.snake_values[snake_rep.snake_id]["head_value"]]
            put_snake_in_frame(self.last_frame, snake_coords, body_color, h_color=head_color, expand_factor=self.expand_factor, offset=self.offset)
            coord_colors[coord_op(snake_rep.body[0], self.offset, '+')] = head_color
            for coord in snake_rep.body[1:]:
                x, y = coord_op(coord, self.offset, '+')
                coord_colors[(x, y)] = body_color
        return [(k, v) for k, v in coord_colors.items()]



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
            frames.append(put_snake_in_frame(self.last_frame.copy(), body_coords, (255, 0, 0), (0, 0, 255), self.expand_factor, self.offset))
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


def put_snake_in_frame(frame, snake_coords, b_color, h_color=None, expand_factor=2, offset=(0, 0)):
    for i, coord in enumerate(snake_coords):
        s_dirs = []
        if i > 0:
            s_dirs.append(coord_op(snake_coords[i-1], coord, '-'))
        if i < len(snake_coords) - 1:
            s_dirs.append(coord_op(snake_coords[i+1], coord, '-'))
        expanded = coord_op(coord, (expand_factor, expand_factor), '*')
        x, y = coord_op(expanded, offset, '+')
        if i == 0 and h_color is not None:
            frame[y, x] = h_color
        else:
            frame[y, x] = b_color
        for s_dir in s_dirs:
            for _ in range(math.ceil(expand_factor / 2) + 1):
                con_x, con_y = coord_op((x, y), s_dir, '+')
                frame[con_y, con_x] = b_color
    return frame
