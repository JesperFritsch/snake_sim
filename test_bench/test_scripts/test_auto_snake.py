import json
import numpy as np
from pathlib import Path
from collections import deque

from snake_sim.snakes.auto_snake import AutoSnake
from snake_sim.utils import Coord
from snake_sim.environment.snake_env import EnvData
from snake_sim.render import core
from snake_sim.render.pygame_render import play_frame_buffer

STATE_FILE_DIR = Path(__file__).parent.parent / 'state_files'
STATE_FILE_NAME = 'state_5273.json'
STATE_FILE_PATH = STATE_FILE_DIR / STATE_FILE_NAME

RUN_STEPS = []

def rgb_color_text(text, r, g, b):
    return f"\033[48;2;{r};{g};{b}m{text}\033[0m"

def test_make_choice(snake: AutoSnake, s_map):
    env_data = EnvData(s_map, {})
    snake.set_new_head = lambda x: print(f"New head: {x}")
    choice = snake.update(env_data.__dict__)
    print(f"Choice: {choice}")

def test_explore(snake: AutoSnake, s_map):
    for tile in snake._valid_tiles(s_map, snake.body_coords[0]):
        result = snake._best_first_search(snake.map.copy(), snake.body_coords.copy(), tile, rundata=RUN_STEPS, safe_margin_factor=snake.SAFE_MARGIN_FACTOR)
        print(f"Tile: {tile}, Result: {result}")
    render_steps(RUN_STEPS)

def test_area_check_direction(snake: AutoSnake, s_map, direction):
    tile = Coord(*snake.coord) + direction
    area_check = snake._area_check_wrapper(s_map, snake.body_coords, tile, safe_margin_factor=snake.SAFE_MARGIN_FACTOR)
    print(f"Direction: {direction}, Area check: {area_check}")

def render_steps(runsteps):
    frames = frame_builder.frames_from_rundata(runsteps)
    play_frame_buffer(frames, grid_width=state_dict['width'], grid_height=state_dict['height'])

def test_area_check(snake: AutoSnake, s_map):
    for tile in snake._valid_tiles(s_map, snake.body_coords[0]):
        area_check = snake._area_check_wrapper(s_map, snake.body_coords, tile, safe_margin_factor=snake.SAFE_MARGIN_FACTOR)
        print(f"Tile: {tile}, Area check: {area_check}")

def run_tests(snake: AutoSnake, s_map):
    # test_make_choice(snake, s_map)
    test_area_check(snake, s_map)
    # test_area_check_direction(snake, s_map, Coord(1, 0))
    test_explore(snake, s_map)

def create_test_snake(id, state_dict, s_map):
    snake = AutoSnake(id, 1, calc_timeout=1500)
    state_dict['snake_values'] = {int(k): v for k, v in state_dict['snake_values'].items()}
    snake.set_init_data(state_dict)
    snake.body_coords = deque([Coord(*coord) for coord in state_dict['snakes'][str(id)]])
    snake._init_area_checker()
    snake.update_map(s_map)
    snake.coord = snake.body_coords[0]
    snake.length = len(snake.body_coords)
    return snake

def print_map(s_map, state_dict, show_snake_id=None):
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

def create_map(state_dict):
    s_map = np.array(state_dict['base_map'], dtype=np.uint8)
    for snake_id, snake in state_dict['snakes'].items():
        body_val = state_dict['snake_values'][snake_id]["body_value"]
        head_val = state_dict['snake_values'][snake_id]["head_value"]
        s_map[snake[0][1]][snake[0][0]] = int(head_val)
        for coord in snake[1:]:
            s_map[coord[1]][coord[0]] = int(body_val)
    for coord in state_dict['food']:
        s_map[coord[1]][coord[0]] = state_dict['food_value']
    return s_map


with open(STATE_FILE_PATH) as state_file:
    state_dict = json.load(state_file)

snake_id = 0
frame_builder = core.FrameBuilder(state_dict, 2, (1, 1))

if snake_id is None:
    snake_items = sorted(state_dict['snakes'].items(), key=lambda x: x[1][0])
    for snake_id, body in snake_items:
        snake_b_val = state_dict['snake_values'][str(snake_id)]["body_value"]
        snake_h_val = state_dict['snake_values'][str(snake_id)]["head_value"]
        print(f"ID: {snake_id: <4} coord: {Coord(*body[0]): <20} body len: {len(body): <4}, body_color: {rgb_color_text('  ', *state_dict['color_mapping'][str(snake_b_val)])}")
else:
    s_map = create_map(state_dict)
    # print_map(s_map, state_dict, snake_id)
    test_snake = create_test_snake(snake_id, state_dict, s_map)
    test_snake.print_map(s_map)
    run_tests(test_snake, s_map)
