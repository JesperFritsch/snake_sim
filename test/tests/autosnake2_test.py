import os
import itertools
from collections import deque
from time import time
from snakes.autoSnake2 import AutoSnake2
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
    test_map_filename = 'test_map_32x32.txt'
    test_map_filepath = os.path.join(test_data_dir, test_map_filename)
    with open(test_map_filepath) as test_map_file:
        map_lines = test_map_file.readlines()
        body_coords = eval(map_lines[0])
        test_map = [[ord(c) for c in row.strip().replace(' ', '')] for row in map_lines[1:]]
    snake = AutoSnake2('C', len(body_coords))
    env.add_snake(snake, (176, 27, 16), (125, 19, 11))
    snake.body_coords = body_coords
    snake.coord = snake.body_coords[0]
    snake.x, snake.y = snake.coord
    snake.map = test_map
    snake.print_map(snake.map)
    print(snake.body_coords)
    snake.pick_direction()
    # snake.get_area_info(snake.map, snake.body_coords, (24,1))
    # check_areas(snake, (0,0))