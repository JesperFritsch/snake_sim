import time
import sys
from io import StringIO
import numpy as np
from typing import List, Dict
from pathlib import Path
import json
from collections import deque
import cProfile
import pstats
import torch

from importlib import resources

from snake_sim.environment.snake_factory import SnakeFactory
from snake_sim.snakes.snake_base import SnakeBase
from snake_sim.environment.snake_strategy_factory import SnakeStrategyFactory
from snake_sim.environment.snake_env import SnakeRep
from snake_sim.environment.types import (
    Coord, 
    DotDict,
    EnvStepData, 
    EnvMetaData, 
    CompleteStepState, 
    StrategyConfig, 
    AreaCheckResult,
    SnakeConfig
)
from snake_sim.debugging import enable_debug_for, activate_debug
from snake_sim.utils import get_locations, profile
from snake_sim.render.utils import create_color_map
from snake_sim.map_utils.general import print_map
from snake_sim.storing.state_storer import load_step_state, get_statefile_dir
from snake_sim.logging_setup import setup_logging
setup_logging(log_level="DEBUG")

with resources.open_text('snake_sim.config', 'default_config.json') as config_file:
    default_config = DotDict(json.load(config_file))

from snake_sim.cpp_bindings.utils import (
    get_dir_to_tile,
    get_visitable_tiles,
    can_make_area_inaccessible
)


def rgb_color_text(text, r, g, b):
    return f"\033[48;2;{r};{g};{b}m{text}\033[0m"


def get_state_file_path():
    state_files = list(get_statefile_dir().glob('*'))
    if not state_files:
        raise FileNotFoundError("No state files found in the directory.")
    latest_file = max(state_files, key=lambda f: f.stat().st_mtime)
    return latest_file


def test_make_choice(snake: SnakeBase, s_map, food_locations: List[Coord] = None):
    print("head coord:", snake._head_coord)
    env_step_data = EnvStepData(s_map, {}, food_locations)
    snake._set_new_head = lambda x: print(f"New head: {x}")
    start_time = time.time()
    for _ in range(1):
        choice = snake.update(env_step_data)
    print(f"Time make choice: {(time.time() - start_time) * 1000}")
    print(f"Choice: {choice}")

def test_recurse_area_check(snake: SnakeBase, s_map, direction=Coord(1,0)):
    tile = Coord(*snake._head_coord) + direction
    start_time = time.time()
    area_check = snake._area_checker.recurse_area_check(
        s_map,
        [tuple(coord) for coord in snake._body_coords],
        (tile.x, tile.y),
        snake._length,
        3,
        # 1.0,
        SnakeBase.SAFE_MARGIN_FRAC + 0.5,
    )
    print(f"Time recurse area check direction {direction}: {(time.time() - start_time) * 1000}")
    print(f"Direction: {direction}, Area check: {area_check}")

def test_area_check_direction(snake: SnakeBase, s_map, direction):
    tile = Coord(*snake._head_coord) + direction
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


def test_area_check(snake: SnakeBase, s_map):
    visitable_tiles = get_visitable_tiles(
        s_map,
        snake._env_meta_data.width,
        snake._env_meta_data.height,
        (snake._body_coords[0].x, snake._body_coords[0].y),
        [snake._env_meta_data.free_value, snake._env_meta_data.food_value]
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
        print(f"AreaCheckResult: {AreaCheckResult(**area_check)}")


def test_area_check_performace(snake: SnakeBase, s_map, iterations=1000, direction=Coord(1,0)):
    stime = time.time()
    tile = Coord(*snake._head_coord) + direction
    for _ in range(iterations):
        area_check = snake._area_check_wrapper(s_map, snake._body_coords, tile, complete_area=True)
    print(f"Time: {(time.time() - stime) * 1000}")
    print(f"Tile: {tile}, Area check: {area_check}")


def test_get_dir_to_tile(snake: SnakeBase, s_map, tile_value, start_coord):
    time_start = time.time()
    direction = get_dir_to_tile(
        s_map,
        snake.env_step_data.width,
        snake.env_step_data.height,
        (start_coord.x, start_coord.y),
        tile_value,
        [snake.env_step_data.free_value, snake.env_step_data.food_value]
    )
    print(f"Time get_dir_to_tile: {(time.time() - time_start) * 1000}")
    print(f"Direction to tile value {tile_value}: {direction}")


def test_get_visitable_tiles(snake: SnakeBase, s_map, center_coord):
    time_start = time.time()
    visitable_tiles = get_visitable_tiles(
        s_map,
        snake.env_step_data.width,
        snake.env_step_data.height,
        (center_coord.x, center_coord.y),
        [snake.env_step_data.free_value, snake.env_step_data.food_value]
    )
    print(f"Time get_visitable_tiles: {(time.time() - time_start) * 1000}")
    print(f"Visitable tiles from {center_coord}: {visitable_tiles}")


def test_spatial_network_ablation(snake: SnakeBase, s_map, food_locations: List[Coord] = None):
    """
    Run spatial ablation analysis using the already created snake, matching the style of other test functions.
    """
    from snake_sim.rl.spatial_network_analyzer import SpatialNetworkAnalyzer
    from snake_sim.rl.state_builder import SnakeContext
    env_step_data = EnvStepData(s_map, {}, food_locations)
    snake_ctx = SnakeContext(
        snake_id=snake._id,
        head=snake._head_coord,
        body_coords=list(snake._body_coords),
        length=snake._length
    )
    rl_state = snake._state_builder.build(
        snake._env_meta_data,
        env_step_data,
        snake_ctx
    )
    # Sanity-check and build tensors
    print("RL state map shape:", rl_state.map.shape)
    print("RL state ctx:", rl_state.ctx)
    map_tensor = torch.from_numpy(rl_state.map).unsqueeze(0).float()  # (1, C, H, W)
    ctx_arr = np.asarray(rl_state.ctx, dtype=np.float32)
    # Ensure ctx is (1, ctx_dim)
    ctx_tensor = torch.from_numpy(ctx_arr.reshape(1, -1)).float()
    action_features = rl_state.meta['action_features']
    print("Action features shape (A,F):", action_features.shape)
    if action_features.ndim == 2:
        action_features = action_features[np.newaxis, ...]  # (1, A, F)
    action_features_tensor = torch.from_numpy(action_features).float()
    analyzer = SpatialNetworkAnalyzer(
        snapshot_dir="models/ppo_training",
        base_name="ppo_model",
    )
    results = analyzer.compare_modes(map_tensor, ctx_tensor, action_features_tensor)
    for mode, res in results.items():
        print(f"[Ablation] Mode: {mode}")
        print("Logits:", res["logits"].detach().cpu().numpy())
        print("Values:", res["values"].detach().cpu().numpy())


# @profile()
def run_tests(snake: SnakeBase, s_map: np.ndarray, step_state: CompleteStepState):
    # test_recurse_area_check(snake, s_map, Coord(1,0))
    test_make_choice(snake, s_map, step_state.food)
    # test_area_check(snake, s_map)
    # test_area_check_performace(snake, s_map, 1000, Coord(0,-1))
    # test_area_check_direction(snake, s_map, Coord(1, 0))
    # test_area_check_direction(snake, s_map, Coord(-1, 0))
    # test_explore(snake, s_map)
    # test_get_dir_to_tile(snake, s_map, snake.env_step_data.food_value, Coord(58, 61))
    # test_get_visitable_tiles(snake, s_map, snake._head_coord)
    test_spatial_network_ablation(snake, s_map, step_state.food)


def create_test_snake(id, snake_reps: Dict[int, SnakeRep], s_map, env_meta_data: EnvMetaData):
    snake: SnakeBase = SnakeFactory().create_snake(
        snake_config=SnakeConfig.from_dict(default_config.snake_config)
        # snake_config=SnakeConfig(
        #     type='survivor',
        #     args={},
        # )
    )
    # snake.set_strategy(1, SnakeStrategyFactory().create_strategy("food_seeker", StrategyConfig("food_seeker")))
    snake.set_id(id)
    snake.set_start_length(1)
    snake_rep = snake_reps[id]
    snake.set_init_data(env_meta_data)
    snake._body_coords = snake_rep.body.copy()
    snake._update_map(s_map)
    snake._head_coord = snake._body_coords[0]
    snake._length = len(snake._body_coords)
    return snake


def print_colored_map(s_map, env_meta_data: EnvMetaData, color_mapping, show_snake_id=None):
    blocked_value = env_meta_data.blocked_value
    highlight_values = env_meta_data.snake_values[show_snake_id].values() if show_snake_id is not None else []
    for row in s_map:
        for val in row:
            if highlight_values:
                color = color_mapping[val] if val in highlight_values or val <= blocked_value else (0x80, 0x80, 0x80)
            else:
                color = color_mapping[val]
            print(rgb_color_text('  ', *color), end='')
        print()


def create_map(step_state: CompleteStepState, snake_reps: Dict[int, SnakeRep]):
    s_map = step_state.env_meta_data.base_map
    for coord in step_state.food:
        s_map[coord[1]][coord[0]] = step_state.env_meta_data.food_value
    for snake_rep in snake_reps.values():
        for coord in snake_rep.body:
            s_map[coord.y, coord.x] = snake_rep.body_value
        s_head = snake_rep.get_head()
        s_map[s_head.y, s_head.x] = snake_rep.head_value
    return s_map


def create_snake_reps(step_state: CompleteStepState) -> Dict[int, SnakeRep]:
    snake_reps = {}
    for snake_id, snake_body in step_state.snake_bodies.items():
        snake_id_int = int(snake_id)
        body_val = step_state.env_meta_data.snake_values[snake_id]["body_value"]
        head_val = step_state.env_meta_data.snake_values[snake_id]["head_value"]
        snake_rep = SnakeRep(snake_id_int, head_val, body_val, Coord(0, 0))
        snake_rep._length = len(snake_body)
        snake_rep.body = deque([Coord(*coord) for coord in snake_body])
        snake_reps[snake_rep.id] = snake_rep
    return snake_reps

def put_food_in_frame(frame, food_coords, color, expand_factor=2, offset=(1, 1)):
    for coord in food_coords:
        coord = (Coord(*coord) * (expand_factor, expand_factor)) + offset
        frame[coord.y, coord.x] = color



if __name__ == "__main__":
    state_file_path = get_state_file_path()
    # state_file_path = r"B:\pythonStuff\snake_sim\test_bench\state_files\state_3628.json"
    step_state: CompleteStepState = load_step_state(state_file_path)
    snake_reps = create_snake_reps(step_state)

    activate_debug()
    enable_debug_for('SnakeBase')
    enable_debug_for('_next_step')
    enable_debug_for('_get_food_dir')
    enable_debug_for('_best_first_search')

    snake_id = 1
    # snake_id = None
    s_map = create_map(step_state, snake_reps)

    if snake_id is None:
        color_mapping = create_color_map(step_state.env_meta_data.snake_values)
        for s_rep in snake_reps.values():
            print(f"ID: {s_rep.id: <4} HEAD: {Coord(*s_rep.get_head()): <20} body len: {s_rep._length: <4}, body_color: {rgb_color_text('  ', *color_mapping[s_rep.body_value])}")
        print_colored_map(s_map, step_state.env_meta_data, color_mapping, show_snake_id=snake_id)

    else:
        test_snake = create_test_snake(snake_id, snake_reps, s_map, step_state.env_meta_data)
        print_map(
            s_map,
            test_snake._env_meta_data.free_value,
            test_snake._env_meta_data.food_value,
            test_snake._env_meta_data.blocked_value,
            test_snake._head_value,
            test_snake._body_value
        )
        # print("snake env_meta_data: ", test_snake._env_meta_data)
        # print("snake env_step_data: ", test_snake._env_step_data)
        # print("snake head: ", test_snake._head_coord)
        # print("snake head value: ", test_snake.get_self_map_values()[0])
        # print("snake body value: ", test_snake.get_self_map_values()[1])
        sys.stdout.flush()
        run_tests(test_snake, s_map, step_state)

        
