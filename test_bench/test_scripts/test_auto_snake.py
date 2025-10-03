import json
import time
import sys
from io import StringIO
import numpy as np
from typing import List, Dict
from pathlib import Path
from collections import deque
import cProfile
import pstats

from importlib import resources

from snake_sim.snakes.survivor_snake import SurvivorSnake
from snake_sim.snakes.strategies.food_strategy import FoodSeeker
from snake_sim.environment.snake_env import SnakeRep
from snake_sim.environment.types import Coord, EnvData, EnvInitData
from snake_sim.render import core
from snake_sim.render.pygame_render_old import play_frame_buffer
from snake_sim.debugging import enable_debug_for, activate_debug
from snake_sim.utils import get_locations, profile
from snake_sim.map_utils.general import print_map

from snake_sim.cpp_bindings.utils import (
    get_dir_to_tile, 
    get_visitable_tiles,
    can_make_area_inaccessible
)

with resources.path('snake_sim', '__init__.py') as init_path:
    STATE_FILE_DIR = Path(init_path.parent) / "test_bench" / "state_files"


RUN_STEPS = []


def rgb_color_text(text, r, g, b):
    return f"\033[48;2;{r};{g};{b}m{text}\033[0m"


def get_state_file_path():
    state_files = list(STATE_FILE_DIR.glob('*'))
    if not state_files:
        raise FileNotFoundError("No state files found in the directory.")
    latest_file = max(state_files, key=lambda f: f.stat().st_mtime)
    return latest_file


def test_make_choice(snake: SurvivorSnake, s_map, food_locations: List[Coord] = None):
    env_data = EnvData(s_map, {}, food_locations)
    snake._set_new_head = lambda x: print(f"New head: {x}")
    start_time = time.time()
    for _ in range(1):
        choice = snake.update(env_data)
    print(f"Time make choice: {(time.time() - start_time) * 1000}")
    print(f"Choice: {choice}")

def test_recurse_area_check(snake: SurvivorSnake, s_map, direction=Coord(1,0)):
    tile = Coord(*snake.get_head_coord()) + direction
    start_time = time.time()
    area_check = snake._area_checker.recurse_area_check(
        s_map,
        [tuple(coord) for coord in snake._body_coords],
        (tile.x, tile.y),
        snake._length,
        3,
        # 1.0,
        SurvivorSnake.SAFE_MARGIN_FRAC + 0.5,
    )
    print(f"Time recurse area check direction {direction}: {(time.time() - start_time) * 1000}")
    print(f"Direction: {direction}, Area check: {area_check}")

def test_explore(snake: SurvivorSnake, s_map):
    for tile in snake._valid_tiles(s_map, snake._body_coords[0]):
        time_start = time.time()
        result = snake._best_first_search(
            snake.map.copy(),
            snake._body_coords.copy(),
            tile,
            rundata=RUN_STEPS,
            exhaustive=True
        )
        print(f"Time explore: {(time.time() - time_start) * 1000}")
        print(f"Tile: {tile}, Result: {result}")


def test_area_check_direction(snake: SurvivorSnake, s_map, direction):
    tile = Coord(*snake.get_head_coord()) + direction
    start_time = time.time()
    for _ in range(1):
        area_check = snake._area_checker.area_check(
            s_map,
            [tuple(coord) for coord in snake._body_coords],
            (tile.x, tile.y),
            10,
            False,
            False,
            True
        )
    print(f"Time area check direction {direction}: {(time.time() - start_time) * 1000}")
    print(f"Direction: {direction}, Coord: {tile} Area check: {area_check}")


def render_steps(runsteps):
    frames = frame_builder.frames_from_rundata(runsteps)
    play_frame_buffer(frames, grid_width=state_dict['width'], grid_height=state_dict['height'])


def test_area_check(snake: SurvivorSnake, s_map):
    visitable_tiles = get_visitable_tiles(
        s_map,
        snake.get_env_init_data().width,
        snake.get_env_init_data().height,
        (snake._body_coords[0].x, snake._body_coords[0].y),
        [snake.get_env_init_data().free_value, snake.get_env_init_data().food_value]
    )
    for tile in visitable_tiles:
        time_start = time.time()
        area_check = snake._area_checker.area_check(
            s_map,
            [tuple(coord) for coord in snake._body_coords],
            (tile[0], tile[1]),
            10,
            False,
            False,
            True
        )
        print(f"Time area check: {(time.time() - time_start) * 1000}")
        print(f"Tile: {tile}, Area check: {area_check}")


def test_area_check_performace(snake: SurvivorSnake, s_map, iterations=1000, direction=Coord(1,0)):
    stime = time.time()
    tile = Coord(*snake.get_head_coord()) + direction
    for _ in range(iterations):
        area_check = snake._area_check_wrapper(s_map, snake._body_coords, tile, complete_area=True)
    print(f"Time: {(time.time() - stime) * 1000}")
    print(f"Tile: {tile}, Area check: {area_check}")


def test_get_dir_to_tile(snake: SurvivorSnake, s_map, tile_value, start_coord):
    time_start = time.time()
    direction = get_dir_to_tile(
        s_map,
        snake.env_data.width,
        snake.env_data.height,
        (start_coord.x, start_coord.y),
        tile_value,
        [snake.env_data.free_value, snake.env_data.food_value]
    )
    print(f"Time get_dir_to_tile: {(time.time() - time_start) * 1000}")
    print(f"Direction to tile value {tile_value}: {direction}")


def test_get_visitable_tiles(snake: SurvivorSnake, s_map, center_coord):
    time_start = time.time()
    visitable_tiles = get_visitable_tiles(
        s_map,
        snake.env_data.width,
        snake.env_data.height,
        (center_coord.x, center_coord.y),
        [snake.env_data.free_value, snake.env_data.food_value]
    )
    print(f"Time get_visitable_tiles: {(time.time() - time_start) * 1000}")
    print(f"Visitable tiles from {center_coord}: {visitable_tiles}")


# @profile()
def run_tests(snake: SurvivorSnake, s_map):
    test_recurse_area_check(snake, s_map, Coord(1,0))
    # test_make_choice(snake, s_map, state_dict['food'])
    test_area_check(snake, s_map)
    # test_area_check_performace(snake, s_map, 1000, Coord(0,-1))
    # test_area_check_direction(snake, s_map, Coord(1, 0))
    # test_area_check_direction(snake, s_map, Coord(-1, 0))
    # test_explore(snake, s_map)
    # test_get_dir_to_tile(snake, s_map, snake.env_data.food_value, Coord(58, 61))
    # test_get_visitable_tiles(snake, s_map, snake.get_head_coord())
    if RUN_STEPS:
        render_steps(RUN_STEPS)


def create_test_snake(id, snake_reps: Dict[int, SnakeRep], s_map, state_dict):
    snake = SurvivorSnake()
    snake.set_strategy(1, FoodSeeker())
    # snake = SurvivorSnake(calc_timeout=1500)
    snake.set_id(id)
    snake.set_start_length(1)
    env_init_data = create_env_init_data(snake_reps.values(), state_dict)
    snake_rep = snake_reps[id]
    snake.set_init_data(env_init_data)
    snake._body_coords = snake_rep.body.copy()
    snake._init_area_checker()
    snake._update_map(s_map)
    snake._head_coord = snake.get_body_coords()[0]
    snake._length = len(snake.get_body_coords())
    return snake


def print_colored_map(s_map, state_dict, show_snake_id=None):
    color_mapping = state_dict['color_mapping']
    blocked_value = state_dict['blocked_value']
    highlight_values = state_dict['snake_values'][str(show_snake_id)].values() if show_snake_id is not None else []
    for row in s_map:
        for val in row:
            if highlight_values:
                color = color_mapping[str(val)] if val in highlight_values or val <= blocked_value else (0x80, 0x80, 0x80)
            else:
                color = color_mapping[str(val)]
            print(rgb_color_text('  ', *color), end='')
        print()


def create_map(state_dict, snake_reps: Dict[int, SnakeRep]):
    s_map = np.array(state_dict['base_map'], dtype=np.uint8)
    for coord in state_dict['food']:
        s_map[coord[1]][coord[0]] = state_dict['food_value']
    for snake_rep in snake_reps.values():
        for coord in snake_rep.body:
            s_map[coord.y][coord.x] = snake_rep.body_value
        s_head = snake_rep.get_head()
        s_map[s_head.y, s_head.x] = snake_rep.head_value
    return s_map


def create_snake_reps(state_dict) -> Dict[int, SnakeRep]:
    snake_reps = {}
    for snake_id, snake in state_dict['snakes'].items():
        snake_id_int = int(snake_id)
        body_val = state_dict['snake_values'][snake_id]["body_value"]
        head_val = state_dict['snake_values'][snake_id]["head_value"]
        snake_rep = SnakeRep(snake_id_int, head_val, body_val, Coord(0, 0))
        snake_rep._length = len(snake)
        snake_rep.body = deque([Coord(*coord) for coord in snake])
        snake_reps[snake_rep.id] = snake_rep
    return snake_reps


def create_env_init_data(snake_reps: List[SnakeRep], state_dict) -> EnvInitData:
    return EnvInitData(
        state_dict['width'],
        state_dict['height'],
        state_dict['free_value'],
        state_dict['blocked_value'],
        state_dict['food_value'],
        {int(k): v for k, v in state_dict['snake_values'].items()},
        {s_rep.id: s_rep.get_head() for s_rep in snake_reps},
        state_dict['base_map']
    )


def put_food_in_frame(frame, food_coords, color, expand_factor=2, offset=(1, 1)):
    for coord in food_coords:
        coord = (Coord(*coord) * (expand_factor, expand_factor)) + offset
        frame[coord.y, coord.x] = color



if __name__ == "__main__":
    STATE_FILE_PATH = get_state_file_path()
    # STATE_FILE_PATH = r"B:\pythonStuff\snake_sim\test_bench\state_files\state_3628.json"
    with open(STATE_FILE_PATH) as state_file:
        state_dict = json.load(state_file)

    snake_reps = create_snake_reps(state_dict)

    activate_debug()
    enable_debug_for('SurvivorSnake')
    enable_debug_for('_next_step')
    enable_debug_for('_get_food_dir')
    enable_debug_for('_best_first_search')

    snake_id = 0
    # snake_id = None
    frame_builder = core.FrameBuilder(state_dict, 2, (1, 1))
    put_food_in_frame(frame_builder.last_frame, state_dict['food'], (0, 255, 0), frame_builder.expand_factor, frame_builder.offset)
    s_map = create_map(state_dict, snake_reps)

    if snake_id is None:
        for s_rep in snake_reps.values():
            print(f"ID: {s_rep.id: <4} HEAD: {Coord(*s_rep.get_head()): <20} body len: {s_rep._length: <4}, body_color: {rgb_color_text('  ', *state_dict['color_mapping'][str(s_rep.body_value)])}")
        print_colored_map(s_map, state_dict, show_snake_id=snake_id)
    else:
        test_snake = create_test_snake(snake_id, snake_reps, s_map, state_dict)
        print_map(
            s_map, 
            test_snake.get_env_init_data().free_value, 
            test_snake.get_env_init_data().food_value, 
            test_snake.get_env_init_data().blocked_value, 
            test_snake.get_self_map_values()[0], 
            test_snake.get_self_map_values()[1]
        )
        print("snake env_init_data: ", test_snake.get_env_init_data())
        print("snake env_data: ", test_snake.get_env_data())
        print("snake head: ", test_snake.get_head_coord())
        print("snake head value: ", test_snake.get_self_map_values()[0],)
        print("snake body value: ", test_snake.get_self_map_values()[1])
        sys.stdout.flush()
        run_tests(test_snake, s_map)
