
import argparse
import numpy as np

import cProfile

from enum import Enum
from PIL import Image
from typing import Dict, Tuple, List
from pathlib import Path
from bitarray import bitarray

from snake_sim.environment.types import Coord

TILESETS_FOLDER = Path(__file__).parent / 'tilesets'

OVERLAP_CHECK_CACHE = {}
DIAG_OVERLAP_CHECK_CACHE = {}


class TileCellValue(Enum):
    FREE = 0
    BLOCKED = 1
    NOCELL = -1


class ImgCellValue(Enum):
    FREE = (255, 255, 255, 255)  # white
    BLOCKED = (0, 0, 0, 255)  # black
    NOCELL = (0, 0, 0, 0)


class BuildTile:
    # tile used in wave function collapse
    def __init__(self, tile_values: np.ndarray):
        self.tile_values = tile_values
        self.height, self.width = tile_values.shape

    def __hash__(self):
        return hash(self.tile_values.tobytes())

    def __eq__(self, other: 'BuildTile'):
        if not isinstance(other, BuildTile):
            raise ValueError("Can only compare BuildTile with another BuildTile")
        return np.array_equal(self.tile_values, other.tile_values)

    def __lt__(self, other: 'BuildTile'):
        return sum(self.tile_values.tobytes()) < sum(other.tile_values.tobytes())

    def rotated(self, k: int) -> 'BuildTile':
        # Rotate the tile 90 degrees clockwise k times
        rotated_values = np.rot90(self.tile_values, -k)
        return BuildTile(rotated_values)

    def flipped(self, axis: int) -> 'BuildTile':
        # Flip the tile along the specified axis (0 for vertical, 1 for horizontal)
        flipped_values = np.flip(self.tile_values, axis=axis)
        return BuildTile(flipped_values)

    def print(self):
        for row in self.tile_values:
            print(' '.join('.' if cell == 0 else 'X' for cell in row))


class Domain:
    def __init__(self, nr_options):
        self._nr_options = nr_options
        self._domain = bitarray(nr_options)
        self._domain.setall(True)

    def collapse_to(self, option_index):
        self._domain.setall(False)
        self._domain[option_index] = True

    def set_options(self, option_indices: List[int]):
        self._domain.setall(False)
        for option_index in option_indices:
            self._domain[option_index] = True

    def remove_option(self, option_index: int):
        self._domain[option_index] = False

    def possible_options(self) -> List[int]:
        return [i for i, bit in enumerate(self._domain) if bit]

    def entropy(self) -> int:
        return self._domain.count()


## Helper functions for _check_compatibility

def _overlapping_blocked(side1: np.ndarray, side2: np.ndarray) -> int:
    # mask of positions where both sides are BLOCKED (from side1's perspective)
    key = (side1.tobytes(), side2.tobytes())
    if key in OVERLAP_CHECK_CACHE:
        return OVERLAP_CHECK_CACHE[key]

    mask = (side1 == TileCellValue.BLOCKED.value) & (side2 == TileCellValue.BLOCKED.value)
    idxs = np.flatnonzero(mask)
    OVERLAP_CHECK_CACHE[key] = (int(idxs.size), idxs.tolist())
    return OVERLAP_CHECK_CACHE[key]

def _diag_overlapping_blocked(side1: np.ndarray, side2: np.ndarray) -> int:
    # diagonal overlaps like this:
    # side1: 1 0 1
    # side2: 0 1 0
    # right-diagonal: side2 element left of side1 position
    key = (side1.tobytes(), side2.tobytes())
    if key in DIAG_OVERLAP_CHECK_CACHE:
        return DIAG_OVERLAP_CHECK_CACHE[key]
    side2_rshift = np.roll(side2, 1)
    side2_rshift[0] = 0  # no wrap-around
    mask_r = (side1 == TileCellValue.BLOCKED.value) & (side2_rshift == TileCellValue.BLOCKED.value)
    idxs_r = np.flatnonzero(mask_r)
    # left-diagonal: side2 element right of side1 position
    side2_lshift = np.roll(side2, -1)
    side2_lshift[-1] = 0  # no wrap-around
    mask_l = (side1 == TileCellValue.BLOCKED.value) & (side2_lshift == TileCellValue.BLOCKED.value)
    idxs_l = np.flatnonzero(mask_l)
    # combine unique indices from side1's perspective
    if idxs_r.size and idxs_l.size:
        idxs = np.unique(np.concatenate((idxs_r, idxs_l)))
    else:
        idxs = idxs_r if idxs_r.size else idxs_l
    DIAG_OVERLAP_CHECK_CACHE[key] = (int(idxs.size), idxs.tolist())
    return DIAG_OVERLAP_CHECK_CACHE[key]


def _any_adjacent_indexes(indexes: List[int]) -> bool:
    indexes = sorted(indexes)
    for i in range(len(indexes) - 1):
        if indexes[i] + 1 == indexes[i + 1]:
            return True
    return False


def _check_side_compatibility(
        outer_side1: np.ndarray,
        inner_side1: np.ndarray,
        outer_side2: np.ndarray,
        inner_side2: np.ndarray
        ) -> bool:
    if sum(outer_side1) == 0 and sum(outer_side2) == 0:
        return True
    overlap_count, overlap_idxs = _overlapping_blocked(outer_side1, outer_side2)
    diag_overlap_count, diag_overlap_idxs = _diag_overlapping_blocked(outer_side1, outer_side2)
    inner_to_outer_overlap1, inner_to_outer_overlap1_idxs = _overlapping_blocked(inner_side1, outer_side2)
    inner_to_outer_overlap2, inner_to_outer_overlap2_idxs = _overlapping_blocked(outer_side1, inner_side2)
    # inner_to_outer_diag1, inner_to_outer_diag1_idxs = _diag_overlapping_blocked(inner_side1, outer_side2)
    # inner_to_outer_diag2, inner_to_outer_diag2_idxs = _diag_overlapping_blocked(outer_side1, inner_side2)
    return (
        not _any_adjacent_indexes(overlap_idxs)
        and (diag_overlap_count == 0 or set(diag_overlap_idxs) == set(overlap_idxs))
        and inner_to_outer_overlap1 <= overlap_count
        and inner_to_outer_overlap2 <= overlap_count
        # and (inner_to_outer_diag1 == 0 or set(inner_to_outer_diag1_idxs) == set(overlap_idxs))
        # and (inner_to_outer_diag2 == 0 or set(inner_to_outer_diag2_idxs) == set(overlap_idxs))
    )


def _check_compatibility(tile1: BuildTile, tile2: BuildTile, side: int) -> bool:
    # tile1 is in the "center", tile2 is the neighbor to check
    # side: 0=top, 1=right, 2=bottom, 3=left
    if side == 0:
        # Check top compatibility
        outer_side1 = tile1.tile_values[0, :]  # top row of tile1
        inner_side1 = tile1.tile_values[1, :]  # second row of tile1
        outer_side2 = tile2.tile_values[-1, :]  # bottom row of tile2
        inner_side2 = tile2.tile_values[-2, :]  # second last row of
    elif side == 1:
        # Check right compatibility
        outer_side1 = tile1.tile_values[:, -1]  # right column of tile1
        inner_side1 = tile1.tile_values[:, -2]  # second last column of tile1
        outer_side2 = tile2.tile_values[:, 0]  # left column of tile2
        inner_side2 = tile2.tile_values[:, 1]  # second column of tile2
    elif side == 2:
        # Check bottom compatibility
        outer_side1 = tile1.tile_values[-1, :]  # bottom row of tile1
        inner_side1 = tile1.tile_values[-2, :]  # second last row of tile1
        outer_side2 = tile2.tile_values[0, :]  # top row of tile2
        inner_side2 = tile2.tile_values[1, :]  # second row of tile2
    elif side == 3:
        # Check left compatibility
        outer_side1 = tile1.tile_values[:, 0]  # left column of tile1
        inner_side1 = tile1.tile_values[:, 1]  # second column of tile1
        outer_side2 = tile2.tile_values[:, -1]  # right column of tile2
        inner_side2 = tile2.tile_values[:, -2]  # second last column of tile2
    else:
        raise ValueError("Side must be 0 (top), 1 (right), 2 (bottom), or 3 (left)")

    return _check_side_compatibility(outer_side1, inner_side1, outer_side2, inner_side2)


def _build_option_mapping(index_to_tile: Dict[int, BuildTile]) -> Dict[BuildTile, Dict[int, bitarray]]:
    option_mapping = {}
    for i, tile1 in index_to_tile.items():
        compatibility = option_mapping.get(tile1, {})
        for j, tile2 in index_to_tile.items():
            for k in range(4):
                bitarr = compatibility.get(k)
                if bitarr is None:
                    bitarr = bitarray(len(index_to_tile))
                    bitarr.setall(False)
                    compatibility[k] = bitarr
                bitarr[j] = _check_compatibility(tile1, tile2, k)
        option_mapping[tile1] = compatibility
    return option_mapping


def _find_build_tile(
    start_pos: Coord,
    img_array: np.ndarray,
    color_mapping: Dict[ImgCellValue, TileCellValue],
    checked_positions: set
) -> BuildTile:
    # Find the tile boundaries
    min_x = min_y = float('inf')
    max_x = max_y = float('-inf')
    to_check = [start_pos]
    while to_check:
        y, x = to_check.pop()
        if (y, x) in checked_positions:
            continue
        checked_positions.add((y, x))
        pixel = tuple(img_array[y, x])
        map_value = color_mapping[pixel]
        if map_value == TileCellValue.NOCELL:
            continue
        min_x = min(min_x, x)
        max_x = max(max_x, x)
        min_y = min(min_y, y)
        max_y = max(max_y, y)
        # Check neighbors (up, down, left, right)
        neighbors = [(y-1, x), (y+1, x), (y, x-1), (y, x+1)]
        for ny, nx in neighbors:
            if 0 <= ny < img_array.shape[0] and 0 <= nx < img_array.shape[1]:
                if (ny, nx) not in checked_positions:
                    to_check.append((ny, nx))

    if min_x == float('inf'):
        # No valid positions found
        return None

    # Extract the tile from the image
    img_cell_array = img_array[min_y:max_y + 1, min_x:max_x + 1]
    tile_cell_array = np.vectorize(lambda x: color_mapping[tuple(x)].value, signature='(n)->()', otypes=[np.uint8])(img_cell_array)
    return BuildTile(tile_cell_array)


def _get_tiles_from_tileset_png(img_path) -> List[BuildTile]:
    img = Image.open(img_path)
    img_array = np.array(img)
    color_mapping = {
        ImgCellValue.FREE.value: TileCellValue.FREE,
        ImgCellValue.BLOCKED.value: TileCellValue.BLOCKED,
        ImgCellValue.NOCELL.value: TileCellValue.NOCELL
    }
    checked_positions = set()
    tiles = []
    for y in range(img_array.shape[0]):
        for x in range(img_array.shape[1]):
            if (y, x) not in checked_positions:
                pixel = tuple(img_array[y, x])
                if color_mapping[pixel] != TileCellValue.NOCELL:
                    tile = _find_build_tile((y, x), img_array, color_mapping, checked_positions)
                    if tile is not None:
                        tiles.append(tile)
                else:
                    checked_positions.add((y, x))
    return tiles


def _generate_all_tile_variations(tiles: List[BuildTile]) -> List[BuildTile]:
    all_variations = set()
    for tile in tiles:
        for k in range(4):  # 4 rotations
            rotated_tile = tile.rotated(k)
            all_variations.add(rotated_tile)
            all_variations.add(rotated_tile.flipped(axis=0))  # vertical flip
            all_variations.add(rotated_tile.flipped(axis=1))  # horizontal flip
    return list(all_variations)


def wave_function_collapse(tiles: List[BuildTile], grid_size: Tuple[int, int]) -> List[List[BuildTile]]:
    tiles = sorted(tiles)
    index_to_tile = {i: tile for i, tile in enumerate(tiles)}
    tile_to_index = {tile: i for i, tile in index_to_tile.items()}
    option_mapping = _build_option_mapping(index_to_tile)

    height, width = grid_size
    grid_domains = [[Domain(len(tiles)) for _ in range(width)] for _ in range(height)]

    # print("Option mapping:")
    # for tile, compatibility in option_mapping.items():
    #     for side, bitarr in compatibility.items():
    #         print("#########")
    #         tile.print()
    #         print(f"  Side {side}: {bitarr}")
    #         print(f"Possible tiles: ")
    #         for i in range(len(bitarr)):
    #             if bitarr[i]:
    #                 index_to_tile[i].print()
    #                 print("")

    # Placeholder for the WFC algorithm implementation
    # This would involve selecting the cell with the lowest entropy,
    # collapsing it to one of its possible tiles, and propagating constraints.

    # For demonstration, we'll just return a grid filled with the first tile
    result_grid = np.full((height, width), tile_to_index[tiles[0]], dtype=int)
    return result_grid


def main(args):
    base_tiles = _get_tiles_from_tileset_png(args.tileset_image)
    all_tiles = _generate_all_tile_variations(base_tiles)
    new_map = wave_function_collapse(all_tiles, (10, 10))
    print(new_map)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process tile images.")
    parser.add_argument("-t", "--tileset-image", type=str, help="Path to the input image file.", default=str(TILESETS_FOLDER / 'tileset1.png'))
    args = parser.parse_args()
    cProfile.run('main(args)', sort='cumtime')
    # main(args)