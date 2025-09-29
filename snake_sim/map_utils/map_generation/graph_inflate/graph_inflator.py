import numpy as np
import random

from itertools import product, permutations
from typing import Tuple, Dict, Union, Set
from networkx import Graph

from snake_sim.map_utils.general import print_map
from snake_sim.environment.types import Coord

class Tile:
    def __init__(self, value: int, coord: Coord):
        self._value = value
        self._coord = coord

    def move(self, direction: Coord):
        assert abs(sum(direction)) == 1, f"Invalid direction: {direction}"
        return self._coord + direction

    def get_neighbor_tiles(self):
        return {d: t for d, t in self._sides.items() if isinstance(t, Tile)}

    def overlaps(self, other: 'Tile'):
        return self._coord == other._coord


class WallTile(Tile):
    def __init__(self, value: int, coord: Coord):
        super().__init__(value, coord)


class DoorWayTile(Tile):
    def __init__(self, value: int, coord: Coord):
        super().__init__(value, coord)


class Area:
    """Represents an area on the map, defined by the outline of WallTiles
    and at least one DoorWayTile.
    """
    def __init__(self, tiles: Set[Tile]):
        self._tiles = tiles
        self._tile_sides: Dict[Tile, Dict[int, Union[Tile, bool]]] = {tile: {} for tile in tiles}
        self._locked_tiles: Set[Tile] = set()

    def add_wall_tile(self, coord: Coord):
        if not self._check_no_overlap(coord):
            raise ValueError(f"Tile at {coord} overlaps with existing tile")
        if not self._check_adjacency(coord):
            raise ValueError(f"Tile at {coord} is not adjacent to existing area")

    def _check_adjacency(self, tile: Tile):
        # check if tile is adjacent to any existing tiles in the area
        return any(abs(sum(tile._coord + other._coord)) == 1 for other in self._tiles)

    def _check_no_overlap(self, tile: Tile):
        return not any(tile.overlaps(other) for other in self._tiles)

    def inflate(self):
        # Inflate the area by marking all tiles within a certain distance as part of the area
        pass

    def get_tiles(self):
        return self._tiles

    def get_locked_tiles(self):
        return self._locked_tiles

    def get_movable_tiles(self):
        return self._tiles - self._locked_tiles

    def lock_overlapping(self, other: 'Area'):
        overlapping = self._tiles.intersection(other.get_tiles())
        self._locked_tiles.update(overlapping)
        other._locked_tiles.update(overlapping)

    def _find_sides(self, tile: Tile):
        neighbors = [Coord(*c) for c in permutations([-1, 0, 1], 2) if abs(sum(c)) == 1]
        for n in neighbors:
            neighbor_coord = tile.move(n)
            neighbor_tile = next((t for t in self._tiles if t._coord == neighbor_coord), None)
            if isinstance(neighbor_tile, WallTile):
                self._tile_sides[tile][n] = neighbor_tile
            elif isinstance(neighbor_tile, DoorWayTile):
                # True for inside area
                self._tile_sides[tile][n] = True
            else:
                # False for outside area
                self._tile_sides[tile][n] = False

    @classmethod
    def create(cls, start_coord: Coord, direction: Coord, free_value: int, blocked_value: int, closed: bool=False):
        neighbors = set(product([-1, 0, 1], repeat=2)) - {(0, 0)}
        if not closed:
            for n in neighbors.copy():
                if direction.x == 0:
                    if n[1] == direction.y:
                        neighbors.remove(n)
                else:
                    if n[0] == direction.x:
                        neighbors.remove(n)
        placed_neighbors = {start_coord + Coord(*n) for n in neighbors}
        tiles = {DoorWayTile(free_value, start_coord)} | {WallTile(blocked_value, c) for c in placed_neighbors}
        instance = cls(tiles)
        for tile in tiles:
            instance._find_sides(tile)
        return instance


def _place_area_on_map(area: Area, map_buffer: np.ndarray):
    for tile in area.get_tiles():
        map_buffer[tile._coord.y, tile._coord.x] = tile._value

def inflate_graph(
        graph: Graph,
        map_shape: Tuple[int, int], # (height, width)
        free_value: int,
        blocked_value: int
    ) -> np.ndarray:
    map_buffer = np.zeros(map_shape, dtype=np.uint8)
    areas = [
        Area.create(
            start_coord=Coord(random.randint(1, map_shape[1]-2), random.randint(1, map_shape[0]-2)),
            direction=Coord(random.choice([-1, 1]), 0),
            free_value=free_value,
            blocked_value=blocked_value,
            closed=False
        ) for _ in range(5)
    ]

    for area in areas:
        _place_area_on_map(area, map_buffer)

    print_map(
        map_buffer,
        free_value=free_value,
        blocked_value=blocked_value,
        food_value=2,
        head_value=3,
        body_value=4,
    )


inflate_graph(Graph(), (20, 40), free_value=0, blocked_value=1)