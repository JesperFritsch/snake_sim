from pathlib import Path

from snake_sim.cpp_bindings.area_check import AreaChecker
from snake_sim.environment.snake_env import SnakeEnv
from snake_sim.environment.food_handlers import FoodHandler
from snake_sim.utils import Coord

# map_path = Path(__file__).parent.parent.parent / "snake_sim" / "maps" / "test_maps" / "test_areacheck.png"
# print(str(map_path))
# env = SnakeEnv(0,0,1,2,0)
# env.set_food_handler(FoodHandler(0, 0, 0))
# env.load_map(map_path)

# env.print_map()
# start_coord = (10, 10)
# AreaChecker(0, 1, 0, 0, env._width, env._height).area_check(env._base_map.flatten(), [(0, 0), (0, 1)], start_coord, 0, True, True, 0.0)

def rot_head_coord(rel_pos, head_pos, direction):
    """
    Transform coordinate to be relative to head_pos and rotated so that direction is "up"
    
    Args:
        rel_pos: Coord object - the position to transform
        head_pos: Coord object - the head position reference point
        direction: Coord object - the direction vector
    
    Returns:
        Coord object with transformed coordinates
    """
    rel_x_abs = abs(rel_pos.x)
    rel_y_abs = abs(rel_pos.y)
    
    return Coord(
        head_pos.x + (rel_x_abs * direction.x) - (rel_y_abs * direction.y),
        head_pos.y + (rel_y_abs * direction.y) - (rel_x_abs * direction.x)
    )

head = Coord(10, 10)
direction = Coord(1, 0)  # up
relative = Coord(-1, 0)

rotated = rot_head_coord(relative, head, direction)
print(rotated)