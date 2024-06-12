import os
import itertools
import cProfile
import pstats
from io import StringIO
import numpy as np
from collections import deque
from pathlib import Path
from time import time
from utils import coord_op, coord_cmp
from snakes.autoSnake2 import AutoSnake2
from snakes.autoSnake3 import AutoSnake3
from snakes.autoSnake4 import AutoSnake4
from snakes.autoSnakeBase import AutoSnakeBase, copy_map
from snake_env import SnakeEnv, RunData

from pygame_render import play_runfile
from video_render import frames_to_video
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
    snake_char = 'W'
    expand_factor = 2
    frame_width = GRID_WIDTH * expand_factor
    frame_height = GRID_HEIGHT * expand_factor
    snake = AutoSnake4(snake_char, 1, calc_timeout=10000)
    frameshape = (frame_width, frame_height, 3)
    base_frame = np.full(frameshape, SnakeEnv.COLOR_MAPPING[SnakeEnv.FREE_TILE], dtype=np.uint8)

    with open(test_map_filepath) as test_map_file:
        step_state = eval(test_map_file.read())
        env.map = env.fresh_map()
        for snake_data in [step_state[s] for s in step_state if s not in (snake_char, 'food')]:
            # print(snake_data)
            core.put_snake_in_frame(base_frame, snake_data, (255, 255, 255), expand_factor=expand_factor)
        for coord in step_state['food']:
            x, y = coord_op(coord, (expand_factor, expand_factor), '*')
            base_frame[y, x] = env.COLOR_MAPPING[SnakeEnv.FOOD_TILE]
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
    # snake.map[22, 9] = ord('X')
    snake.print_map(snake.map)
    # print(snake.body_coords)
    print(snake.length, snake.coord, snake.body_value)
    print(snake.coord)
    # snake.update()
    # frames = None
    rundata = []
    area_coord = (15, 16)
    map_copy = snake.map.copy()
    map_copy[area_coord[1], area_coord[0]] = ord('Q')
    snake.print_map(map_copy)


    # pr = cProfile.Profile()
    # pr.enable()

    # # # Call the function you want to profile
    # for _ in range(200):
    #     snake.area_check(snake.map, snake.body_coords.copy(), area_coord)

    # pr.disable()

    # # Print the profiling results
    # s = StringIO()
    # sortby = 'cumulative'
    # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # ps.print_stats()
    # print(s.getvalue())


    # time_e = time()
    # for _ in range(200):
    #     area = snake.area_check(snake.map, snake.body_coords.copy(), area_coord)
    # print(area)
    # print('area_check: ', (time() - time_e) * 1000)
    # time_e = time()
    # for _ in range(2000):
    #     areas = snake.get_areas_fast(snake.map, area_coord)
    # print('get_areas_fast: ', (time() - time_e) * 1000)
    # time_e = time()
    # for _ in range(200000):
    #     a = area_coord[0] == area_coord[0] and area_coord[1] == area_coord[1]
    # print('manual: ', (time() - time_e) * 1000)
    # time_e = time()
    # for _ in range(200000):
    #     a = coord_cmp(area_coord, area_coord)
    # print('coord_cmp: ', (time() - time_e) * 1000)
    # print(areas)
    # time_e = time()
    # for _ in range(2000):
    #     is_single = snake.is_single_area2(snake.map, area_coord, coord_op(area_coord, (1, 0), '+'))
    # print(is_single, coord_op(area_coord, (1, 0), '+'))
    # print('is_single_area2: ', (time() - time_e) * 1000)
    # valid_tiles = snake.valid_tiles(snake.map, area_coord)
    # time_z = time()
    # areas_check = snake.is_area_clear(snake.map, snake.body_coords, area_coord)
    # print(areas_check)
    # print(f"Time is_area_clear: {(time() - time_z) * 1000}")
    # time_z = time()
    # areas_check = snake.area_check(snake.map, snake.body_coords, area_coord)
    # print(areas_check)
    # print(f"Time area_check: {(time() - time_z) * 1000}")

    pr = cProfile.Profile()
    pr.enable()

    for tile in snake.valid_tiles(snake.map, snake.coord):
        planned_path = None
        # planned_path = snake.get_route(snake.map, tile , target_tiles=list(env.food.locations))
        # print(f"Planned path: {planned_path}")
        # print(snake.check_safe_food_route(snake.map, planned_path))
        # snake.print_map(snake.map)
        if planned_path:
            tile = planned_path.pop()
        # planned_path = None
        s_time = time()
        option = snake.deep_look_ahead(snake.map.copy(), tile, snake.body_coords.copy(), snake.length, rundata=rundata, planned_route=planned_path)
        print('free_path: ', option['free_path'])
        print(f"Time: {(time() - s_time) * 1000}")
    frames = []

    pr.disable()

    # Print the profiling results
    s = StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())

    for body_coords in rundata:
        frame = core.put_snake_in_frame(base_frame.copy(), body_coords, (255, 0, 0), expand_factor=expand_factor)
        frames.append(frame)
        # frames.append(frame)
    play_runfile(frames=frames, grid_width=frame_width, grid_height=frame_width)
    # video_output = Path(__file__).parent.joinpath('..', '..', 'render', 'videos', 'test_look_ahead.mp4').resolve()
    # frames_to_video(frames, str(video_output), 30, size=(640, 640))
