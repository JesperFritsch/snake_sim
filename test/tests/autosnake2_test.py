import os
import itertools
from collections import deque
from time import time
from snakes.autoSnake2 import AutoSnake2
from snakes.autoSnake3 import AutoSnake3
from snakes.autoSnake4 import AutoSnake4
from snakes.autoSnakeBase import AutoSnakeBase, copy_map
from snake_env import SnakeEnv

def check_areas(snake: AutoSnake2, coord):
    valid_tiles = snake.valid_tiles(snake.map, coord)
    s_time = time()
    print(f"areas for {coord}: {snake.get_areas(snake.map, coord)}")
    print(f"Time: {(time() - s_time) * 1000}")

if __name__ == '__main__':
    GRID_WIDTH = 32
    GRID_HEIGHT = 32
    FOOD = 35
    env = SnakeEnv(GRID_WIDTH, GRID_HEIGHT, FOOD)
    test_data_dir = os.path.join(os.getcwd(), 'test', 'test_data')
    test_map_filename = 'test_map2.txt'
    test_map_filepath = os.path.join(test_data_dir, test_map_filename)
    with open(test_map_filepath) as test_map_file:
        map_lines = test_map_file.readlines()
        body_coords = eval(map_lines[0])
        test_map = []
        for row in map_lines[1:]:
            map_row = []
            for c in row.strip().replace(' ', ''):
                if c == '.':
                    map_row.append(env.FREE_TILE)
                elif c == 'F':
                    map_row.append(env.FOOD_TILE)
                else:
                    map_row.append(ord(c))
            test_map.append(map_row)

    snake = AutoSnake4('C', len(body_coords))
    env.add_snake(snake, (176, 27, 16), (125, 19, 11))
    snake.body_coords = body_coords
    snake.coord = snake.body_coords[0]
    snake.x, snake.y = snake.coord
    snake.map = test_map
    s_time = time()
    route = snake.get_route(snake.map, (20, 31), (20, 15))
    print('Time: ', (time() - s_time) * 1000)
    # snake.map[30][20] = ord('X')
    print(snake.valid_tiles(snake.map, (20, 31)))
    print('route: ', route)
    map_copy = copy_map(snake.map)
    snake.show_route(map_copy, route)
    print(snake.verify_route(route))
    snake.print_map(map_copy)
    # snake.print_map(snake.map)
    # snake.pick_direction()
    # print(snake.is_area_clear(snake.map, snake.body_coords, (16, 25)))
    # snake.get_area_info(snake.map, snake.body_coords, (24,1))
    # check_areas(snake, (0,0))