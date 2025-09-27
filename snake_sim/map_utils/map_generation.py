
import argparse
import numpy as np

from enum import Enum
from PIL import Image
from typing import Dict, Tuple, List
from pathlib import Path

from snake_sim.environment.types import Coord

TILESETS_FOLDER = Path(__file__).parent / 'tilesets'


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
    
    def rotated(self, k: int) -> 'BuildTile':
        # Rotate the tile 90 degrees clockwise k times
        rotated_values = np.rot90(self.tile_values, -k)
        return BuildTile(rotated_values)

    def flipped(self, axis: int) -> 'BuildTile':
        # Flip the tile along the specified axis (0 for vertical, 1 for horizontal)
        flipped_values = np.flip(self.tile_values, axis=axis)
        return BuildTile(flipped_values)


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
                    if tile:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process tile images.")
    parser.add_argument("-t", "--tileset-image", type=str, help="Path to the input image file.", default=str(TILESETS_FOLDER / 'tileset1.png'))
    args = parser.parse_args()

    base_tiles = _get_tiles_from_tileset_png(args.tileset_image)
    all_tiles = _generate_all_tile_variations(base_tiles)
    for i, tile in enumerate(all_tiles):
        print(f"Tile {i}:")
        print(tile.tile_values)
        print()