
from snake_sim.environment.types import Coord

ACTION_ORDER = {
    Coord(0, -1): 0,  # Up
    Coord(1, 0): 1,   # Right
    Coord(0, 1): 2,   # Down
    Coord(-1, 0): 3   # Left
}

ACTION_ORDER_INVERSE = {v: k for k, v in ACTION_ORDER.items()}