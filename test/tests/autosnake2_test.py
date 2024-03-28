import os
import itertools
from time import time
from snakes.autoSnake2 import AutoSnake2
from snake_env import SnakeEnv

def check_areas(snake: AutoSnake2, coord):
    valid_tiles = snake.valid_tiles(snake.map, coord)
    s_time = time()
    print(f"areas for {coord}: {snake.get_areas(snake.map, coord, valid_tiles)}")
    print(f"Time: {(time() - s_time) * 1000}")

if __name__ == '__main__':
    GRID_WIDTH = 32
    GRID_HEIGHT = 32
    FOOD = 35
    snake = AutoSnake2('A', 50)
    env = SnakeEnv(GRID_WIDTH, GRID_HEIGHT, FOOD)
    env.add_snake(snake, (176, 27, 16), (125, 19, 11))
    test_data_dir = os.path.join(os.getcwd(), 'test', 'test_data')
    test_map_filename = 'test_map_32x32.txt'
    test_map_filepath = os.path.join(test_data_dir, test_map_filename)
    with open(test_map_filepath) as test_map_file:
        map_lines = test_map_file.readlines()
        test_map = [[ord(c) for c in row.strip().replace(' ', '')] for row in map_lines[1:]]

    snake.map = test_map
    snake.print_map(snake.map)

    check_areas(snake, (1, 30))