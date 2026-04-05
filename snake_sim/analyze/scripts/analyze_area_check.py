from pathlib import Path

from snake_sim.cpp_bindings.area_check import AreaChecker
from snake_sim.cpp_bindings.utils import get_visitable_tiles
from snake_sim.environment.types import AreaCheckResult
from snake_sim.environment.snake_env import SnakeEnv
from snake_sim.environment.food_handlers import FoodHandler
from snake_sim.utils import Coord

map_path = Path(__file__).parent.parent.parent / "snake_sim" / "maps" / "test_maps" / "test_areacheck.png"
map_path = Path("~/Downloads/image.png").expanduser()
print(str(map_path))
env = SnakeEnv(0,0,1,2,0)
env.set_food_handler(FoodHandler(0, 0, 0))
env.load_map(map_path)

env.print_map()
s_map = env._base_map.flatten() 
env_meta_data = env.get_init_data()
devide_coord = (15, 32)
visitable_values = [env_meta_data.free_value, env_meta_data.food_value]
visitable_tiles = get_visitable_tiles(
    s_map,
    env._width,
    env._height,
    devide_coord,
    visitable_values
)
for tile in visitable_tiles:
    edited_map = env._base_map.copy()
    edited_map[devide_coord[1], devide_coord[0]] = env_meta_data.blocked_value
    result = AreaChecker(0, 1, 0, 0, env._width, env._height).area_check(edited_map.flatten(), [(0, 0), (0, 1)], tile, 0, False, True, True)
    check_result = AreaCheckResult(**result)
    print(tile, check_result)

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