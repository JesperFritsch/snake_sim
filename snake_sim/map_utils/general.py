import sys
import numpy as np
import json
from typing import Dict, Tuple
from itertools import permutations
from importlib import resources
from PIL import Image
from snake_sim.environment.types import Coord, DotDict

with resources.open_text('snake_sim.config', 'default_config.json') as config_file:
    default_config = DotDict(json.load(config_file))

def get_map_files_mapping():
    files = list(resources.files('snake_sim.maps.map_images').iterdir())
    mapping = {f.name.split('.')[0]: f for f in files if f.is_file()}
    mapping.pop('__init__')
    return mapping

def get_maps_info():
    maps_mapping = get_map_files_mapping()
    maps_info = {}
    for map_name, file in maps_mapping.items():
        np_map = convert_png_to_map(
            file,
            color_mapping={
                (0, 0, 0, 0): default_config.free_value,
                (255, 0, 0, 255): default_config.food_value,
                (0, 0, 0, 255): default_config.blocked_value,
            }
        )

        maps_info[map_name] = {
            'height': np_map.shape[0],
            'width': np_map.shape[1],
        }
    return maps_info

def rgb_color_text(text, r, g, b):
    return f"\033[48;2;{r};{g};{b}m{text}\033[0m"

def print_map(
        s_map: np.ndarray,
        free_value: int,
        food_value: int,
        blocked_value: int,
        head_value: int=None,
        body_value: int=None,
        color_map: Dict[int, Tuple[int, int, int]]={}
    ) -> int: # number of lines printed
    width, height = s_map.shape
    max_nr_digits_width = len(str(width))
    max_nr_digits_height = len(str(height))
    w_nr_strings = [str(i).rjust(max_nr_digits_width) for i in range(height)]
    h_nr_strings = [str(i).rjust(max_nr_digits_height) for i in range(width)]
    digit_rows = [' '.join([f"{nr_string[i]}" for nr_string in w_nr_strings]) for i in range(max_nr_digits_width)]
    map_rows = []
    for i, row in enumerate(s_map):
        map_row = [h_nr_strings[i]]
        for c_i, c in enumerate(row):
            c_color = color_map.get(c)
            if c == free_value:
                if i % 2 == c_i % 2:
                    map_row.append('.')
                else:
                    map_row.append(',')
            elif c == food_value:
                if c_color:
                    map_row.append(rgb_color_text(' ', *c_color))
                else:
                    map_row.append('F')
            elif c == blocked_value:
                map_row.append('#')
            elif c == head_value:
                if c_color:
                    map_row.append(rgb_color_text(' ', *c_color))
                else:
                    map_row.append(f'A')
            elif c == body_value:
                if c_color:
                    map_row.append(rgb_color_text(' ', *c_color))
                else:
                    map_row.append('a')
            elif c % 2 == 0:
                if c_color:
                    map_row.append(rgb_color_text(' ', *c_color))
                else:
                    map_row.append(f'b')
            else:
                if c_color:
                    map_row.append(rgb_color_text(' ', *c_color))
                else:
                    map_row.append(f'H')
        map_row.append(h_nr_strings[i])
        map_rows.append(' '.join(map_row))
    def _safe_write(s: str):
        # Attempt write; if BlockingIOError, retry a few short times
        import time
        attempts = 0
        while attempts < 5:
            try:
                sys.stdout.write(s)
                return
            except BlockingIOError:
                time.sleep(0.01)
                attempts += 1
        # Final attempt without swallow; may raise
        sys.stdout.write(s)

    for digit_row in digit_rows:
        _safe_write(' ' * (max_nr_digits_height + 1) + digit_row + '\n')
    for row in map_rows:
        _safe_write(row + '\n')
    for digit_row in digit_rows:
        _safe_write(' ' * (max_nr_digits_height + 1) + digit_row  + '\n')
    try:
        sys.stdout.flush()
    except BlockingIOError:
        # Retry flush briefly
        import time
        for _ in range(5):
            time.sleep(0.01)
            try:
                sys.stdout.flush()
                break
            except BlockingIOError:
                continue
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

def expand_map(
        s_map: np.ndarray,
        expansion: int,
        free_value: int,
        blocked_value: int,
    ) -> np.ndarray:
    """ Scale a value map up by `expansion`, keeping blocked areas connected.

    Cell (x, y) lands on pixel (x * expansion, y * expansion). Pixels between two
    adjacent blocked cells are filled in as well, so walls stay solid instead of
    turning into dotted lines.
    """
    height, width = s_map.shape
    neighbors = [Coord(*c) for c in permutations([-1, 0, 1], 2) if abs(sum(c)) == 1]
    expanded_map = np.full(
        ((height * expansion) - (expansion - 1), (width * expansion) - (expansion - 1)),
        free_value,
        dtype=s_map.dtype
    )
    for y in range(height):
        for x in range(width):
            coord = Coord(x, y)
            if s_map[y, x] == blocked_value:
                ex_coord = coord * expansion
                expanded_map[ex_coord.y, ex_coord.x] = blocked_value
                if expansion > 1:
                    # if the expand factor is greater than 1, we need to color the neighbors of the blocked cell in the map
                    for n in neighbors:
                        n_coord = coord + n
                        if 0 <= n_coord.x < width and 0 <= n_coord.y < height and s_map[n_coord.y, n_coord.x] == blocked_value:
                            ex_n_coord = ex_coord + n
                            expanded_map[ex_n_coord.y, ex_n_coord.x] = blocked_value
    return expanded_map
