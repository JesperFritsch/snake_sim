import random
import string
import os
import shutil
import json
from time import time

from collections import deque

class DotDict(dict):
    def __init__(self, other_dict):
        for k, v in other_dict.items():
            if isinstance(v, dict):
                v = DotDict(v)
            self[k] = v

    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError:
            raise AttributeError()

    def __setattr__(self, attr, value):
        self[attr] = value

    def __delattr__(self, attr):
        try:
            del self[attr]
        except KeyError:
            raise AttributeError


def exec_time(func):
    def wrapper(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        print(f"Execution time for {func.__name__}: {round((time() - start) * 1000, 3)} ms")
        return result
    return wrapper

def coord_cmp(coord1, coord2):
    return coord1[0] == coord2[0] and coord1[1] == coord2[1]

def coord_op(coord_left, coord_right, op):
    # Check the operation and perform it directly
    if op == '+':
        return tuple(l + r for l, r in zip(coord_left, coord_right))
    elif op == '-':
        return tuple(l - r for l, r in zip(coord_left, coord_right))
    elif op == '*':
        return tuple(l * r for l, r in zip(coord_left, coord_right))
    else:
        raise ValueError("Unsupported operation")


def rand_str(n):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=n))


def get_run_step(filename, step_nr):
    with open(filename) as run_json:
        run_dict = json.load(run_json)
        steps = run_dict['steps']
        final_step = max([int(k) for k in steps.keys()])
        current = 1
        coords_map = {}
        last_step = steps['1']
        while current <= step_nr < final_step:
            step_data = steps[str(current)]
            coords_map['food'] = step_data['food']
            snake_data = step_data['snakes']
            last_snakes = last_step['snakes']
            for snake in snake_data:
                body = coords_map.get(snake['snake_id'], deque())
                curr_head = snake['curr_head']
                body.appendleft(curr_head)
                last_snake = max(last_snakes, key=lambda x: x['snake_id'] == snake['snake_id'])
                if last_snake['tail_dir'] != [0, 0]:
                    body.pop()
                coords_map[snake['snake_id']] = body
            current += 1
            last_step = step_data
    return coords_map


if __name__ == '__main__':
    from render import core
    run_dir = os.path.join(os.getcwd(), 'runs', 'batch', 'grid_32x32')
    rpi_runs = os.path.join(run_dir, 'rpi')
    for file in os.listdir(run_dir):
        nr_snakes, text, grid, randstr, steps, empty = file.split('_')
        new_filename = f'{nr_snakes}_snakes_rpi_{randstr}_{steps}.run'
        pixel_changes = core.pixel_changes_from_runfile(os.path.join(run_dir, file))
        pixel_change_list = pixel_changes['changes']
        with open(os.path.join(rpi_runs, new_filename), 'w') as f:
            for change in pixel_change_list:
                f.write(json.dumps(change) + '\n')
