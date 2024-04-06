import random
import string
import os
import shutil
import json

from render import core


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


def get_body_coords_from_run_step(filename, snake_id, step):
    grid_changes = core.grid_changes_from_runfile(filename)
    step_changes = grid_changes['changes']


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
