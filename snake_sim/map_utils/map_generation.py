
import argparse
import numpy as np

import cProfile

from enum import Enum
from PIL import Image
from typing import Dict, Tuple, List
from pathlib import Path
from bitarray import bitarray

from snake_sim.map_utils.general import print_map
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
        self.shape = tile_values.shape

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

    def copy(self) -> 'Domain':
        """Return a deep copy of this Domain."""
        new = Domain(self._nr_options)
        new._domain = self._domain.copy()
        return new

    def remove_option(self, option_index: int):
        self._domain[option_index] = False

    def possible_options(self) -> List[int]:
        return [i for i, bit in enumerate(self._domain) if bit]

    def entropy(self) -> int:
        return self._domain.count()

    def intersect_with_bitarray(self, mask: bitarray) -> bool:
        """Intersect domain with mask (bitarray). Return True if domain changed."""
        if len(mask) != self._nr_options:
            raise ValueError("Mask length does not match domain size")
        old = self._domain.copy()
        self._domain &= mask
        return old != self._domain

    def is_collapsed(self) -> bool:
        return self.entropy() == 1


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
    inner_overlap_count, inner_overlap_idxs = _overlapping_blocked(inner_side1, inner_side2)
    diag_overlap_count1, diag_overlap_idxs1 = _diag_overlapping_blocked(outer_side1, outer_side2)
    diag_overlap_count2, diag_overlap_idxs2 = _diag_overlapping_blocked(outer_side2, outer_side1)
    inner_to_outer_overlap1, inner_to_outer_overlap1_idxs = _overlapping_blocked(inner_side1, outer_side2)
    inner_to_outer_overlap2, inner_to_outer_overlap2_idxs = _overlapping_blocked(outer_side1, inner_side2)
    inner_to_outer_diag1, inner_to_outer_diag1_idxs = _diag_overlapping_blocked(inner_side1, outer_side2)
    inner_to_outer_diag2, inner_to_outer_diag2_idxs = _diag_overlapping_blocked(outer_side1, inner_side2)
    return (
        not _any_adjacent_indexes(overlap_idxs)
        and not _any_adjacent_indexes(inner_overlap_idxs)
        and (diag_overlap_count1 == 0 or (
                set(diag_overlap_idxs1) == set(overlap_idxs)
                or set(diag_overlap_idxs2) == set(overlap_idxs)
            )
        )
        and inner_to_outer_overlap1 <= overlap_count
        and inner_to_outer_overlap2 <= overlap_count
        and (inner_to_outer_diag1 == 0 or set(inner_to_outer_diag1_idxs) == set(overlap_idxs))
        and (inner_to_outer_diag2 == 0 or set(inner_to_outer_diag2_idxs) == set(overlap_idxs))
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

    result = _check_side_compatibility(outer_side1, inner_side1, outer_side2, inner_side2)
    # print("###########\n")
    # print(f"side: {side}")
    # tile1.print()
    # print()
    # tile2.print()
    # print(f"compatible: {result}\n")
    return result


def _build_option_mapping(idx_to_tile: Dict[int, BuildTile]) -> Dict[int, Dict[int, bitarray]]:
    # Build a compatibility mapping keyed by tile index for faster lookup during propagation
    n = len(idx_to_tile)
    option_mapping: Dict[int, Dict[int, bitarray]] = {}
    for i, tile1 in idx_to_tile.items():
        compatibility: Dict[int, bitarray] = {}
        for k in range(4):
            bitarr = bitarray(n)
            bitarr.setall(False)
            compatibility[k] = bitarr
        option_mapping[i] = compatibility

    for i, tile1 in idx_to_tile.items():
        for j, tile2 in idx_to_tile.items():
            for k in range(4):
                option_mapping[i][k][j] = _check_compatibility(tile1, tile2, k)

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


def _generate_all_tile_variations(tiles: List[BuildTile]) -> Dict[int, BuildTile]:
    all_variations = set()
    for tile in tiles:
        for k in range(4):  # 4 rotations
            rotated_tile = tile.rotated(k)
            all_variations.add(rotated_tile)
            all_variations.add(rotated_tile.flipped(axis=0))  # vertical flip
            all_variations.add(rotated_tile.flipped(axis=1))  # horizontal flip
    return {i: tile for i, tile in enumerate(sorted(list(all_variations)))}


def _in_bounds(pos: Coord, height: int, width: int) -> bool:
    x, y = pos
    return 0 <= y < height and 0 <= x < width


def _neighbors(pos: Coord, height: int, width: int) -> List[Tuple[Coord, int]]:
    x, y = pos
    possible = [
        (Coord(x, y - 1), 0),
        (Coord(x + 1, y), 1),
        (Coord(x, y + 1), 2),
        (Coord(x - 1, y), 3)
    ]
    return [p for p in possible if _in_bounds(p[0], height, width)]


def _calculate_gridsize_and_offset(map_shape: Tuple[int, int], tile_shape: Tuple[int, int]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    map_height, map_width = map_shape
    tile_height, tile_width = tile_shape
    if map_height < 2 * tile_height or map_width < 2 * tile_width:
        raise ValueError("Map dimensions must be at least 2 times tile dimensions")
    grid_height = map_height // tile_height
    grid_width = map_width // tile_width
    offset_y = (map_height - (grid_height * tile_height)) // 2
    offset_x = (map_width - (grid_width * tile_width)) // 2
    return (grid_height, grid_width), (offset_y, offset_x)


def _assemble_build_tiles(
        tile_grid: np.ndarray,
        idx_to_tile: Dict[int, BuildTile],
        tile_shape: Tuple[int, int],
        spacing: int = 0,
        spacing_value: int = 255
    ) -> np.ndarray:
    tile_grid_height, tile_grid_width = tile_grid.shape
    tile_height, tile_width = tile_shape
    spacing_height = spacing * (tile_grid_height - 1)
    spacing_width = spacing * (tile_grid_width - 1)
    buffer = np.full((
        tile_grid_height * tile_height + spacing_height,
        tile_grid_width * tile_width + spacing_width
    ), spacing_value, dtype=np.uint8)
    for grid_y in range(tile_grid_height):
        for grid_x in range(tile_grid_width):
            tile_index = tile_grid[grid_y, grid_x]
            tile = idx_to_tile[tile_index]
            start_y = grid_y * tile_height + spacing * grid_y
            start_x = grid_x * tile_width + spacing * grid_x
            buffer[start_y:start_y + tile_height, start_x:start_x + tile_width] = tile.tile_values
    return buffer

def _assemble_map_from_tiles(
        tile_grid: np.ndarray,
        idx_to_tile: Dict[int, BuildTile],
        tile_shape: Tuple[int, int],
        map_shape: Tuple[int, int],
        offset: Tuple[int, int],
        free_value: int = TileCellValue.FREE.value,
        blocked_value: int = TileCellValue.BLOCKED.value,
        spacing: int = 0
    ) -> np.ndarray:
    map_height, map_width = map_shape
    offset_y, offset_x = offset
    final_map = np.full((map_height, map_width), TileCellValue.FREE.value, dtype=np.uint8)
    assembled_tiles = _assemble_build_tiles(tile_grid, idx_to_tile, tile_shape)
    # put assembled tiles into final map with offset
    final_map[offset_y:offset_y + assembled_tiles.shape[0], offset_x:offset_x + assembled_tiles.shape[1]] = assembled_tiles
    # replace tile cell values with desired free/blocked values
    final_map[final_map == TileCellValue.FREE.value] = free_value
    final_map[final_map == TileCellValue.BLOCKED.value] = blocked_value
    return final_map

# Helpers for wfc


def _get_lowest_entropy_cell(grid: List[List[Domain]], rng: np.random.Generator) -> Coord:
    min_entropy = None
    min_positions: List[Coord] = []
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0
    for y in range(height):
        for x in range(width):
            e = grid[y][x].entropy()
            if e > 1:
                if min_entropy is None or e < min_entropy:
                    min_entropy = e
                    min_positions = [Coord(x, y)]
                elif e == min_entropy:
                    min_positions.append(Coord(x, y))
    return min_positions[rng.integers(0, len(min_positions))]

def _check_contradiction(grid: List[List[Domain]]) -> bool:
    for row in grid:
        for cell in row:
            if cell.entropy() == 0:
                return True
    return False


def _all_cells_collapsed(grid: List[List[Domain]]) -> bool:
    for row in grid:
        for cell in row:
            if cell.entropy() != 1:
                return False
    return True


def _check_consistency(grid: List[List[Domain]], option_mapping: Dict[int, Dict[int, bitarray]]) -> bool:
    for y in range(len(grid)):
        for x in range(len(grid[0])):
            domain = grid[y][x]
            if domain.entropy() == 0:
                return False
            for (ny, nx), side in _neighbors((y, x), len(grid), len(grid[0])):
                neighbor = grid[ny][nx]
                if not neighbor.is_consistent_with(domain, side, option_mapping):
                    return False
    return True

def _backtrack(grid: List[List[Domain]], stack: List[List[List[Domain]]]) -> Tuple[List[List[Domain]], bool]:
    while stack:
        grid = stack.pop()
        if not _check_contradiction(grid):
            return grid, True
    return grid, False


def _propagate(queue: List[Coord], grid_domains, option_mapping) -> bool:
    height = len(grid_domains)
    width = len(grid_domains[0]) if height > 0 else 0
    while queue:
        d_coord = queue.pop(0)
        domain: Domain = grid_domains[d_coord.y][d_coord.x]
        domain_options = domain.possible_options()
        for n_coord, side in _neighbors(d_coord, height, width):
            neighbor: Domain = grid_domains[n_coord.y][n_coord.x]
            allowed = bitarray(neighbor._nr_options)
            allowed.setall(False)
            for t_idx in domain_options:
                allowed |= option_mapping[t_idx][side]
            changed = neighbor.intersect_with_bitarray(allowed)
            if changed:
                if neighbor.entropy() == 0:
                    return False  # contradiction
                queue.append(n_coord)
    return True


def _get_collapse_choice(cell_to_collapse: Domain, rng: np.random.Generator):
    options = cell_to_collapse.possible_options()
    return options[rng.integers(0, len(options))]


def _copy_grid(grid: List[List[Domain]]) -> List[List[Domain]]:
    return [[cell.copy() for cell in row] for row in grid]


def _wave_function_collapse(
        idx_to_tile: Dict[int, BuildTile],
        grid_size: Tuple[int, int],
        option_mapping: Dict[int, Dict[int, bitarray]],
        seed: int = None,
    ) -> np.ndarray:

    height, width = grid_size
    n_tiles = len(idx_to_tile)
    grid_domains: List[List[Domain]] = [[Domain(n_tiles) for _ in range(width)] for _ in range(height)]

    rng = np.random.default_rng(seed)

    stack = []

    while True:

        if _all_cells_collapsed(grid_domains):
            break

        cell_to_collapse: Coord = _get_lowest_entropy_cell(grid_domains, rng)
        x = cell_to_collapse.x
        y = cell_to_collapse.y
        domain = grid_domains[y][x]
        choice = _get_collapse_choice(domain, rng)

        state_snapshot = _copy_grid(grid_domains)
        stack.append((state_snapshot))

        # print(f"collapsing cell {(y, x)} with options {domain.possible_options()} to {choice}")
        # tile1 = idx_to_tile[choice]
        # neighbor1 =

        domain.collapse_to(choice)
        if not _propagate([Coord(x, y)], grid_domains, option_mapping):
            grid_state, successful = _backtrack(grid_domains, stack)
            grid_domains = grid_state
            if not successful:
                raise RuntimeError("Wave Function Collapse failed: no valid tiling found with given tiles and constraints")

    if not _all_cells_collapsed(grid_domains):
        raise RuntimeError("Cell not collapsed at end of WFC")

    result = np.zeros((height, width), dtype=int)
    for y in range(height):
        for x in range(width):
            opts = grid_domains[y][x].possible_options()
            result[y, x] = opts[0]

    return result


def print_tile_grid(idx_to_tile: Dict[int, BuildTile], grid: np.ndarray, tile_shape: Tuple[int, int]):
    assembled_tiles = _assemble_build_tiles(
        grid,
        idx_to_tile,
        tile_shape=tile_shape,
        spacing=1
    )
    print_map(
        assembled_tiles,
        free_value=TileCellValue.FREE.value,
        blocked_value=TileCellValue.BLOCKED.value,
        food_value=255,
        head_value=255,
        body_value=255,
    )


def print_option_mapping(option_mapping: Dict[int, Dict[int, bitarray]], idx_to_tile: Dict[int, BuildTile]):
    for tile_idx, sides in option_mapping.items():
        tile = idx_to_tile[tile_idx]
        print(f"Tile {tile_idx} (shape {tile.shape}):")
        tile.print()
        for side, bitarr in sides.items():
            options = [i for i, bit in enumerate(bitarr) if bit]
            side_name = ['top', 'right', 'bottom', 'left'][side]
            print(f"Tile {tile_idx} (shape {tile.shape}):")
            print(f"  {side_name}: {options}")
            print()
            for opt in options:
                idx_to_tile[opt].print()
                print()
        print()




def main(args):
    base_tiles = _get_tiles_from_tileset_png(args.tileset_image)
    idx_to_tile = _generate_all_tile_variations(base_tiles)
    tile_shape = list(idx_to_tile.values())[0].shape
    grid_shape, offset = _calculate_gridsize_and_offset((args.height, args.width), tile_shape)
    option_mapping = _build_option_mapping(idx_to_tile)

    # print_option_mapping(option_mapping, idx_to_tile)

    tile_grid = _wave_function_collapse(idx_to_tile, grid_shape, option_mapping, seed=args.seed)

    print(tile_grid)

    assembled_map = _assemble_map_from_tiles(tile_grid, idx_to_tile, tile_shape, (args.height, args.width), offset)
    print_map(
        assembled_map,
        free_value=TileCellValue.FREE.value,
        blocked_value=TileCellValue.BLOCKED.value,
        food_value=255,
        head_value=255,
        body_value=255,
    )
    print_tile_grid(
        idx_to_tile,
        grid=tile_grid,
        tile_shape=tile_shape
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process tile images.")
    parser.add_argument("-t", "--tileset-image", type=str, help="Path to the input image file.", default=str(TILESETS_FOLDER / 'tileset1.png'))
    parser.add_argument("-s", "--seed", type=int, help="Random seed (deterministic map)", default=10)
    parser.add_argument("-W", "--width", type=int, help="Map width in tiles", default=32)
    parser.add_argument("-H", "--height", type=int, help="Map height in tiles", default=32)
    args = parser.parse_args()
    # cProfile.run('main(args)', sort='cumtime')
    main(args)