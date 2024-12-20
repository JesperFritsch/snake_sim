import numpy as np
import os
import json
from typing import List, Dict
from collections import deque
from pathlib import Path

from snake_sim.protobuf import sim_msgs_pb2
from snake_sim.utils import coord_cmp, rand_str as rand_str_generator, Coord

class StepData:
    def __init__(self, food: list, step: int) -> None:
        self.snakes: List[dict] = []
        self.food = food
        self.step = step
        self.full_state = True

    @classmethod
    def from_dict(cls, step_dict):
        step_data = cls(food=step_dict['food'], step=step_dict['step'])
        step_data.full_state = all([snake.get('body') for snake in step_dict['snakes']])
        for snake_data in step_dict['snakes']:
            step_data.snakes.append(snake_data)
        return step_data

    @classmethod
    def from_protobuf(cls, step_data):
        step = cls(food=[(f.x, f.y) for f in step_data.food], step=step_data.step)
        for snake in step_data.snakes:
            snake_data = {
                'snake_id': snake.snake_id,
                'curr_head': (snake.curr_head.x, snake.curr_head.y),
                'prev_head': (snake.prev_head.x, snake.prev_head.y),
                'curr_tail': (snake.curr_tail.x, snake.curr_tail.y),
                'head_dir': (snake.head_dir.x, snake.head_dir.y),
                'did_eat': snake.did_eat,
                'did_turn': snake.did_turn,
                'did_grow': snake.did_grow,
                'body': []
            }
            if snake.body:
                snake_data['body'] = [(c.x, c.y) for c in snake.body]
            else:
                step.full_state = False
            step.snakes.append(snake_data)
        return step

    def add_snake_data(self, snake_id: int, snake_coords: List[Coord], did_grow: bool = False):
        head_dir = snake_coords[0] - snake_coords[1]
        did_eat = False
        turn = None
        if snake_coords[0] in self.food:
            did_eat = True
        last_dir = snake_coords[1] + snake_coords[2] if len(snake_coords) > 2 else head_dir
        if last_dir != head_dir:
            if last_dir == (0, -1):
                if head_dir == (1, 0):
                    turn = 'right'
                else:
                    turn = 'left'
            elif last_dir == (1, 0):
                if head_dir == (0, 1):
                    turn = 'right'
                else:
                    turn = 'left'
            elif last_dir == (0, 1):
                if head_dir == (-1, 0):
                    turn = 'right'
                else:
                    turn = 'left'
            elif last_dir == (-1, 0):
                if head_dir == (0, -1):
                    turn = 'right'
                else:
                    turn = 'left'

        base_snake_data = {
            'snake_id': snake_id,
            'curr_head': snake_coords[0],
            'prev_head': snake_coords[1] if len(snake_coords) > 1 else snake_coords[0],
            'curr_tail': snake_coords[-1],
            'head_dir': head_dir,
            'did_eat': did_eat,
            'did_turn': turn,
            'did_grow': did_grow,
            'body': snake_coords
        }
        self.snakes.append(base_snake_data)

    def to_dict(self, full_state=False):
        if full_state:
            return self.__dict__
        else:
            snakes = [snake.copy() for snake in self.snakes]
            for snake in snakes:
                snake['body'] = []
            not_full_state = self.__dict__.copy()
            not_full_state['snakes'] = snakes
            not_full_state['full_state'] = False
            return not_full_state

    def to_protobuf(self, full_state=False):
        step_data = sim_msgs_pb2.StepData()
        step_data.step = self.step
        step_data.full_state = full_state
        for snake_data in self.snakes:
            snake = step_data.snakes.add()
            snake.snake_id = snake_data['snake_id']
            snake.curr_head.x, snake.curr_head.y = snake_data['curr_head']
            snake.prev_head.x, snake.prev_head.y = snake_data['prev_head']
            snake.curr_tail.x, snake.curr_tail.y = snake_data['curr_tail']
            snake.head_dir.x, snake.head_dir.y = snake_data['head_dir']
            snake.did_eat = snake_data['did_eat']
            snake.did_grow = snake_data['did_grow']
            if snake_data['did_turn']:
                snake.did_turn = snake_data['did_turn']
            if full_state:
                for coord in snake_data['body']:
                    body_coord = snake.body.add()
                    body_coord.x, body_coord.y = coord
        for coord in self.food:
            food = step_data.food.add()
            food.x, food.y = coord
        return step_data


class RunData:
    def __init__(self,
                width: int,
                height: int,
                snakes: list,
                base_map: np.array,
                food_value: int,
                free_value: int,
                blocked_value: int,
                color_mapping: dict,
                record_file: str = None):  # New parameter added
        self.width = width
        self.height = height
        self.snakes = snakes
        self.base_map = base_map
        self.food_value = food_value
        self.free_value = free_value
        self.blocked_value = blocked_value
        self.color_mapping = color_mapping
        self.steps: Dict[int, StepData] = {}
        self.record_file = record_file  # New attribute added

    def get_metadata(self):
        metadata = self.to_dict()
        del metadata['steps']
        return metadata

    @classmethod
    def from_dict(cls, run_dict):
        run_data = cls(
            width=run_dict['width'],
            height=run_dict['height'],
            snakes=run_dict['snakes'],
            base_map=np.array(run_dict['base_map'], dtype=np.uint8),
            color_mapping={int(k): v for k, v in run_dict['color_mapping'].items()},
            record_file=run_dict.get('record_file')  # New field handled
        )
        for step_nr, step_dict in run_dict['steps'].items():
            step_data_obj = StepData.from_dict(step_dict)
            run_data.add_step(int(step_nr), step_data_obj)
        run_data.food_value = run_dict['food_value']
        run_data.free_value = run_dict['free_value']
        run_data.blocked_value = run_dict['blocked_value']
        run_data.color_mapping = {int(k): v for k, v in run_dict['color_mapping'].items()}
        return run_data

    @classmethod
    def from_json_file(cls, filepath):
        with open(filepath) as file:
            return cls.from_dict(json.load(file))

    @classmethod
    def from_protobuf(cls, run_data):
        meta_data = run_data.run_meta_data
        run = cls(
            width=meta_data.width,
            height=meta_data.height,
            snakes=meta_data.snakes,
            base_map=np.frombuffer(bytes(meta_data.base_map), dtype=np.uint8).reshape(meta_data.height, meta_data.width)
        )
        run.color_mapping.update({int(k): (v.r, v.g, v.b) for k, v in meta_data.color_mapping.items()})
        for step_nr, step in run_data.steps.items():
            step_data = StepData.from_protobuf(step)
            run.add_step(step_nr, step_data)
        return run

    @classmethod
    def from_protobuf_file(cls, filepath):
        run_data = sim_msgs_pb2.RunData()
        with open(filepath, 'rb') as file:
            run_data.ParseFromString(file.read())
        return cls.from_protobuf(run_data)

    def add_step(self, step: StepData):
        self.steps[step.step] = step

    def get_coord_mapping(self, step_nr):
        steps = self.steps
        final_step = len(steps)
        current = 1
        coords_map = {}
        last_step: StepData = steps[current]
        current_step: StepData = None
        while current <= step_nr < final_step:
            current_step = steps[current]
            snakes = current_step.snakes
            for snake in snakes:
                body = coords_map.setdefault(snake['snake_id'], deque([snake['prev_head']]))
                curr_head = snake['curr_head']
                body.appendleft(curr_head)
                if last_step is not None:
                    last_snake_step = list(filter(lambda x: x['snake_id'] == snake['snake_id'], last_step.snakes))[0]
                    if not coord_cmp(last_snake_step['curr_tail'], snake['curr_tail']):
                        body.pop()
            current += 1
            last_step = current_step
        if current_step is not None:
            coords_map['food'] = current_step.food
            return coords_map
        return None

    def write_to_file(self, output_dir, aborted=False, ml=False, filename=None, as_proto=False):
        runs_dir = Path(output_dir)
        run_dir = os.path.join(runs_dir, f'grid_{self.width}x{self.height}')
        if filename is None:
            os.makedirs(run_dir, exist_ok=True)
            aborted_str = '_ABORTED' if aborted else ''
            grid_str = f'{self.width}x{self.height}'
            nr_snakes = f'{len(self.snakes)}'
            rand_str = rand_str_generator(6)
            file_type = "pb" if as_proto else "json"
            filename = f'{nr_snakes}_snakes_{grid_str}_{rand_str}_{len(self.steps)}{"_ml_" if ml else ""}{aborted_str}.{file_type}'
        filepath = Path(os.path.join(run_dir, filename))
        as_proto = as_proto or filepath.suffix == '.pb'

        print(f"saving run data to '{filepath}'")
        if as_proto:
            run_data = self.to_protobuf()
            with open(filepath, 'wb') as file:
                file.write(run_data.SerializeToString())
        else:
            with open(filepath, 'w') as file:
                json.dump(self.to_dict(), file)

    def to_dict(self):
        return {
            'width': self.width,
            'height': self.height,
            'food_value': self.food_value,
            'free_value': self.free_value,
            'blocked_value': self.blocked_value,
            'color_mapping': self.color_mapping,
            'snakes': self.snakes,
            'base_map': self.base_map.tolist(),
            'steps': {k: v.to_dict() for k, v in self.steps.items()},
            'record_file': self.record_file  # New field added
        }

    def to_protobuf(self, full_state=False):
        run_data = sim_msgs_pb2.RunData()
        metadata = run_data.run_meta_data
        metadata.width = self.width
        metadata.height = self.height
        metadata.food_value = self.food_value
        metadata.free_value = self.free_value
        metadata.blocked_value = self.blocked_value
        for key, value in self.color_mapping.items():
            metadata.color_mapping[key].r = value[0]
            metadata.color_mapping[key].g = value[1]
            metadata.color_mapping[key].b = value[2]
        for snake_id in self.snakes:
            metadata.snake_ids.add(snake_id)
        for step_nr, step_data in self.steps.items():
            step_data = step_data.to_protobuf(full_state)
            run_data.steps[step_nr].CopyFrom(step_data)
        metadata.base_map.extend(self.base_map.ravel().tolist())
        metadata.record_file = self.record_file  # New field added
        return run_data