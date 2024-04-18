import os
import itertools
from collections import deque
from time import time
from utils import coord_op
from snakes.autoSnake2 import AutoSnake2
from snakes.autoSnake3 import AutoSnake3
from snakes.autoSnake4 import AutoSnake4
from snakes.autoSnakeBase import AutoSnakeBase, copy_map
from snake_env import SnakeEnv, RunData

from pygame_render import play_runfile
from render import core

def check_areas(snake, coord):
    valid_tiles = snake.valid_tiles(snake.map, coord)
    s_time = time()
    print(f"areas for {coord}: {snake.get_areas(snake.map, coord)}")
    print(f"Time: {(time() - s_time) * 1000}")

if __name__ == '__main__':
    GRID_WIDTH = 32
    GRID_HEIGHT = 32
    FOOD = 35
    env = SnakeEnv(GRID_WIDTH, GRID_HEIGHT, FOOD)
    test_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'test_data'))
    test_map_filename = 'test_map2.txt'
    test_map_filepath = os.path.join(test_data_dir, test_map_filename)
    snake_char = 'G'
    expand_factor = 2
    frame_width = GRID_WIDTH * expand_factor
    frame_height = GRID_HEIGHT * expand_factor
    snake = AutoSnake4(snake_char, 1)
    base_frame = [SnakeEnv.COLOR_MAPPING[SnakeEnv.FREE_TILE]] * (frame_width * frame_height)

    with open(test_map_filepath) as test_map_file:
        step_state = eval(test_map_file.read())
        env.map = env.fresh_map()
        for snake_data in [step_state[s] for s in step_state if s not in (snake_char, 'food')]:
            print(snake_data)
            core.put_snake_in_frame(base_frame, frame_width, snake_data, (255, 255, 255), expand_factor=expand_factor)
        for coord in step_state['food']:
            x, y = coord_op(coord, (expand_factor, expand_factor), '*')
            base_frame[y * frame_width + x] = env.COLOR_MAPPING[SnakeEnv.FOOD_TILE]
        for key, coords in step_state.items():
            if key == snake_char:
                value = snake.body_value
            elif key == 'food':
                value = SnakeEnv.FOOD_TILE
            else:
                value = ord('x')
            env.put_coords_on_map(coords, value)
    snake_head = step_state.get(snake_char)[0]
    if snake_head is None:
        raise ValueError("Snake head not found in test map")
    env.add_snake(snake, (176, 27, 16), (125, 19, 11))
    snake.body_coords = deque([tuple(c) for c in step_state.get(snake_char)])
    snake.length = len(snake.body_coords)
    snake.coord = snake.body_coords[0]
    env.food.locations = set([tuple(x) for x in step_state['food'] if tuple(x) != snake.coord])
    snake.x, snake.y = snake.coord
    snake.update_map(env.map)
    # snake.print_map(snake.map)
    # print(snake.body_coords)
    print(snake.length, snake.coord, snake.body_value)
    print(snake.coord)
    # snake.update()
    # frames = None
    rundata = []
    for tile in snake.valid_tiles(snake.map, snake.coord):
        planned_path = snake.get_route(snake.map, tile , target_tiles=list(env.food.locations))
        print(f"Planned path: {planned_path}")
        if planned_path:
            tile = planned_path.pop()
        # planned_path = None
        s_time = time()
        option = snake.deep_look_ahead(copy_map(snake.map), tile, snake.body_coords.copy(), snake.length, rundata=rundata, planned_route=planned_path)
        print(option)
        print(f"Time: {(time() - s_time) * 1000}")
    frames = []
    for body_coords in rundata:
        frame = core.put_snake_in_frame(base_frame.copy(), frame_width, body_coords, (255, 0, 0), expand_factor=expand_factor)
        frames.append(frame)
    play_runfile(frames=frames, grid_width=frame_width, grid_height=frame_width)