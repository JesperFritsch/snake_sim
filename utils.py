import random
import string
import os
import shutil
import json


def rand_str(n):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=n))

if __name__ == '__main__':
    run_dir = os.path.join(os.getcwd(), 'runs', 'grid_32x32')
    for file in os.listdir(run_dir):
        name, end = os.path.splitext(file)
        comps = name.split('_')
        if len(comps) != 4:
            continue
        s_nr, text, grid, rand = comps
        with open(os.path.join(run_dir, file)) as f:
            run_data = json.load(f)
            steps = len(run_data['steps'])
            new_name = '_'.join([str(x) for x in [s_nr, text, grid, rand, steps]]) + end
        shutil.move(os.path.join(run_dir, file), os.path.join(run_dir, new_name))
        print(f"renamed {os.path.join(run_dir, file)} to {os.path.join(run_dir, new_name)}")

def generate_scenario():
    body_coords = []
    map_rows = []
    map_row = []

    for y in range(32):
        for x in range(32):
            if x != x_line and y != y_line:
                map_row.append(0)
        print()