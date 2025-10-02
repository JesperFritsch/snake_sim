import sys
import numpy as np
from typing import Dict, Tuple
from importlib import resources
from PIL import Image

def get_map_files_mapping():
    files = list(resources.files('snake_sim.maps.map_images').iterdir())
    mapping = {f.name.split('.')[0]: f for f in files if f.is_file()}
    mapping.pop('__init__')
    return mapping


def print_map(s_map: np.ndarray, free_value: int, food_value: int, blocked_value: int, head_value: int, body_value: int) -> int: # number of lines printed
    width, height = s_map.shape
    max_nr_digits_width = len(str(width))
    max_nr_digits_height = len(str(height))
    w_nr_strings = [str(i).rjust(max_nr_digits_width) for i in range(height)]
    h_nr_strings = [str(i).rjust(max_nr_digits_height) for i in range(width)]
    digit_rows = [' '.join([f"{nr_string[i]}" for nr_string in w_nr_strings]) for i in range(max_nr_digits_width)]
    map_rows = []
    for i, row in enumerate(s_map):
        map_row = [h_nr_strings[i]]
        for c in row:
            if c == free_value:
                map_row.append('.')
            elif c == food_value:
                map_row.append('F')
            elif c == blocked_value:
                map_row.append('#')
            elif c == head_value:
                map_row.append(f'A')
            elif c == body_value:
                map_row.append('a')
            elif c % 2 == 0:
                map_row.append(f'b')
            else:
                map_row.append(f'H')
        map_row.append(h_nr_strings[i])
        map_rows.append(' '.join(map_row))
    for digit_row in digit_rows:
        print(' ' * (max_nr_digits_height + 1) + digit_row)
    for row in map_rows:
        print(row)
    for digit_row in digit_rows:
        print(' ' * (max_nr_digits_height + 1) + digit_row)
    sys.stdout.flush()
    return len(digit_rows) * 2 + len(map_rows)


def convert_png_to_map(
        file_path: str, 
        color_mapping: Dict[Tuple[int, int, int, int], int], # RGBA to map value
    ) -> np.ndarray:

    img = Image.open(file_path)
    img_array = np.array(img)
    _validate_map_values(img_array, color_mapping)
    vectorized_map = np.vectorize(lambda x: color_mapping[tuple(x)], signature='(n)->()', otypes=[np.uint8])
    s_map = vectorized_map(img_array)
    return s_map


def _validate_map_values(map_array: np.ndarray, color_mapping: Dict[Tuple[int, int, int, int], int]):
    valid_values = list(color_mapping.keys())
    invalid_mask = ~np.isin(map_array, valid_values)
    
    if np.any(invalid_mask):
        invalid_positions = np.where(invalid_mask)
        invalid_values = np.unique(map_array[invalid_mask])
        positions = list(zip(invalid_positions[0], invalid_positions[1]))
        
        raise ValueError(
            f"Map contains invalid values {invalid_values} not in color_mapping. "
            f"Found at positions: {positions[:5]}{'...' if len(positions) > 5 else ''}"
        )