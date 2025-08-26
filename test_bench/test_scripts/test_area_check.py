from pathlib import Path

from snake_sim.cpp_bindings.area_check import AreaChecker
from snake_sim.environment.snake_env import SnakeEnv
from snake_sim.environment.food_handlers import FoodHandler
from snake_sim.utils import Coord

map_path = Path(__file__).parent.parent.parent / "snake_sim" / "maps" / "test_maps" / "test_areacheck.png"
print(str(map_path))
env = SnakeEnv(0,0,1,2,0)
env.set_food_handler(FoodHandler(0, 0, 0))
env.load_map(map_path)

env.print_map()
start_coord = (10, 10)
AreaChecker(0, 1, 0, 0, env._width, env._height).area_check(env._base_map.flatten(), [(0, 0), (0, 1)], start_coord, 0, True, True, 0.0)


