import os
import itertools
import cProfile
import pstats
import sys
from io import StringIO
import numpy as np
from collections import deque
from pathlib import Path
from time import time
from pprint import pprint
from snake_sim.utils import coord_op, coord_cmp
from snake_sim.snakes.auto_snake import AutoSnake
from snake_sim.snake_env import SnakeEnv, RunData

from snake_sim.cpp_bindings.area_check import AreaChecker

from snake_sim.render.pygame_render import play_frame_buffer
from snake_sim.render.video_render import frames_to_video
from snake_sim.render import core

def check_areas(snake, coord):
    valid_tiles = snake.valid_tiles(snake.map, coord)
    s_time = time()
    print(f"areas for {coord}: {snake.get_areas(snake.map, coord)}")
    print(f"Time: {(time() - s_time) * 1000}")

if __name__ == '__main__':
    GRID_WIDTH = 128
    GRID_HEIGHT = 128

    FOOD = 15
    expand_factor = 2
    offset = (1, 1)
    env = SnakeEnv(GRID_WIDTH, GRID_HEIGHT, FOOD)
    snake_map = 'comps2'
    # test_map = r"B:\pythonStuff\snake_sim\snake_sim\maps\test_maps\testmap3.png"
    env.load_png_map(snake_map)

    # env.load_png_map(test_map)
    env.init_recorder()
    frame_builder = core.FrameBuilder(env.run_data.to_dict(), expand_factor, offset)

    test_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'test_data'))
    test_map_filename = 'test_map1.txt'
    test_map_filepath = os.path.join(test_data_dir, test_map_filename)
    snake_char = 'A'
    frame_width = env.width * expand_factor
    frame_height = env.height * expand_factor
    snake = AutoSnake(snake_char, 1, calc_timeout=1500)
    env.add_snake(snake, (255, 255, 255), (0, 0, 0))
    frameshape = (frame_width, frame_height, 3)
    base_frame = frame_builder.last_frame

    with open(test_map_filepath) as test_map_file:
        step_state = eval(test_map_file.read())
    # step_state = {
    #     "food": [],
    #     "A": deque([(0, 0) for _ in range(200)])
    # }
    # pprint(step_state)

    env.map = env.fresh_map()
    for snake_data in [step_state[s] for s in step_state if s not in (snake_char, 'food')]:
        core.put_snake_in_frame(base_frame, snake_data, (255, 255, 255), h_color=(0, 0, 255), expand_factor=expand_factor, offset=offset)
    for coord in step_state['food']:
        expanded = coord_op(coord, (expand_factor, expand_factor), '*')
        x, y = coord_op(expanded, offset, '+')
        base_frame[y, x] = env.COLOR_MAPPING[SnakeEnv.FOOD_TILE]
    env.put_coords_on_map(step_state['food'], SnakeEnv.FOOD_TILE)
    for key, coords in step_state.items():
        if key == 'food':
            continue
        print(coords[0], key)
        env.put_coords_on_map([coords[0]], ord(key))
        value = ord(key.lower())
        coords = [c for c in coords if c != coords[0]]
        env.put_coords_on_map(coords, value)
    for id in step_state:
        if id != 'food':
            print("ID: ", id, len(step_state[id]))
            if id != snake_char:
                s = AutoSnake(id, 1)
                env.add_snake(s, (176, 27, 16), (125, 19, 11))
            else:
                s = env.snakes[id]
            s.body_coords = deque([tuple(c) for c in step_state.get(id)])
            s.length = len(s.body_coords)
            s.coord = s.body_coords[0]
            s.x, s.y = s.coord
            s.update_map(env.map)
    env.food.locations = set([tuple(x) for x in step_state['food'] if tuple(x) != snake.coord])
    # snake.map[22, 9] = ord('X')
    # snake.print_map(snake.map)
    env.print_map(env.map)
    sys.stdout.flush()
    # print(snake.body_coords)
    print(snake.length, snake.coord, snake.body_value)
    print(snake.coord)
    # snake.update()
    # frames = None
    rundata = []
    # area_coord = (22,27)
    ac = AreaChecker(env.FOOD_TILE, env.FREE_TILE, snake.body_value, env.width, env.height)
    snake.update_env_data(env._get_env_data(snake.id))
    snake.update_map(snake.env_data.map)

    # time_e = time()
    # print(snake.length)
    # for _ in range(1000):
    #     area = ac.area_check(snake.map, list(snake.body_coords), coord_op(snake.coord, (0, 1), '+'), False)
    # execution_time = (time() - time_e) * 1000
    # print('area_check: ', execution_time)
    # print(area)

    pr = cProfile.Profile()
    pr.enable()

    stime = time()
    choice = snake._pick_direction()
    print(f"Choice: {choice}")
    print(f"snake.coord: {snake.coord}")
    print(snake)
    print(f"Time: {(time() - stime) * 1000}")

    print("is_single: ", ac.is_single_entrance(snake.map, (28,11), (28,12)))

    for tile in snake._valid_tiles(snake.map, snake.coord):
    #     s_time = time()
    #     risk = snake.calc_immediate_risk(env.map, tile, 3)
    #     print(f"Time: {(time() - s_time) * 1000}")
    #     print(f"Risk: {risk}")

        # s_time = time()
        # option = snake._deep_look_ahead(snake.map.copy(), tile, snake.body_coords.copy(), snake.length, rundata=rundata)
        # print(f'_deep_look_ahead: {tile}', option['free_path'])
        # print(f"Time: {(time() - s_time) * 1000}")

        s_time = time()
        option = snake._best_first_search(snake.map.copy(), snake.body_coords.copy(), tile, rundata=rundata, safe_margin_factor=snake.SAFE_MARGIN_FACTOR)
        print(f'_best_first_search: {tile}', option)
        print(f"Time: {(time() - s_time) * 1000}")

        s_time = time()
        area_check = snake._area_check_wrapper(snake.map, snake.body_coords.copy(), tile, food_check=False, exhaustive=False, safe_margin_factor=0.08)
        print(f"area_check for tile {tile}: {area_check}")
        print(f"Time: {(time() - s_time) * 1000}")

    # # print(snake.get_future_available_food_map())
    # s_time = time()
    # head_dist = (0, 1)
    # start_coord = coord_op(snake.coord, head_dist, '+')
    # # start_coord = 0, 28
    # # area_check = snake._area_check_wrapper(snake.map, snake.body_coords.copy(), start_coord, food_check=False, exhaustive=False, target_margin=100)
    # # print(f"area_check for tile {coord_op(snake.coord, head_dist, '+')}: {area_check}")

    # # option = snake._deep_look_ahead(snake.map.copy(), coord_op(snake.coord, head_dist, '+'), snake.body_coords.copy(), snake.length, rundata=rundata)
    # option = snake._best_first_search(snake.map.copy(), snake.body_coords.copy(), coord_op(snake.coord, head_dist, '+'), rundata=rundata)
    # print(f"area_check for tile {head_dist}: {option}")
    # print(f"Time: {(time() - s_time) * 1000}")
    # time_e = time()
    # area = snake._area_check_wrapper(snake.map, snake.body_coords, start_coord)
    # execution_time = (time() - time_e) * 1000
    # print('area_check: ', execution_time)
    # print(area)
    # frames = []

    pr.disable()

    # Print the profiling results
    s = StringIO()
    sortby = 'tottime'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # ps.print_stats()
    # print(s.getvalue())

    frames = frame_builder.frames_from_rundata(rundata)
    play_frame_buffer(frames, grid_width=frame_width, grid_height=frame_width)
    # video_output = Path(__file__).parent.joinpath('..', '..', 'render', 'videos', 'test_look_ahead.mp4').resolve()
    # frames_to_video(frames, str(video_output), 30, size=(640, 640))
