import os
import itertools
from collections import deque
from time import time
from utils import coord_op
from snakes.autoSnake2 import AutoSnake2
from snakes.autoSnake3 import AutoSnake3
from snakes.autoSnake4 import AutoSnake4
from snakes.autoSnakeBase import AutoSnakeBase, copy_map
from snake_env import SnakeEnv

from pygame_render import playback_runfile

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
    snake_char = 'C'
    with open(test_map_filepath) as test_map_file:
        map_lines = test_map_file.readlines()
        # body_coords = eval(map_lines[0])
        snake_head = None
        test_map = []
        for y, row in enumerate(map_lines[1:]):
            map_row = []
            for x, c in enumerate(row.strip().replace(' ', '')):
                if c == '.':
                    map_row.append(env.FREE_TILE)
                elif c == 'F':
                    map_row.append(env.FOOD_TILE)
                else:
                    if c == snake_char:
                        snake_head = (x, y)
                    map_row.append(ord(c))
            test_map.append(map_row)
    if snake_head is None:
        raise ValueError("Snake head not found in test map")
    snake = AutoSnake4(snake_char, 1)
    env.add_snake(snake, (176, 27, 16), (125, 19, 11))
    snake.body_coords = snake.find_body_coords(test_map, snake_head)
    snake.length = len(snake.body_coords)
    snake.coord = snake.body_coords[0]
    snake.x, snake.y = snake.coord
    snake.map = test_map
    print(snake.length, snake.coord)
    # s_time = time()
    frames = []
    new_tile = coord_op(snake.coord, (0, 1), '+')
    snake.deep_look_ahead(snake.map, new_tile, snake.body_coords, snake.length, frames=frames)
    # print(frames)
    playback_runfile(frames=frames, grid_width=GRID_WIDTH, grid_height=GRID_HEIGHT)
    # snake.print_map(snake.map)
    # snake.pick_direction()
    # print(snake.is_area_clear(snake.map, snake.body_coords, (16, 25)))
    # snake.get_area_info(snake.map, snake.body_coords, (24,1))
    # check_areas(snake, (0,0))