import os
import sys
import datetime
import json
import argparse
from multiprocessing import Pipe, Process
from pathlib import Path

from .utils import DotDict
from .snake_env import SnakeEnv
from .snakes.autoSnake4 import AutoSnake4
from .render.pygame_render import play_runfile, play_stream
config = None

def setup_env(config):
    env = SnakeEnv(config.grid_width, config.grid_height, config.food, config.food_decay)
    if config.get('map'):
        env.load_png_map(config.map)
    count = 0
    for snake_config in config.snake_configs:
        count += 1
        env.add_snake(AutoSnake4(**snake_config['snake'], calc_timeout=config.calc_timeout), **snake_config['env'])
        if count == config.snake_count:
            break
    return env


def handle_args(args, config):

    if args.grid_width:
        config.grid_width = args.grid_width
    if args.grid_height:
        config.grid_height = args.grid_height
    if args.food:
        config.food = args.food
    if args.snake_count:
        config.snake_count = args.snake_count
    if args.calc_timeout:
        config.calc_timeout = args.calc_timeout
    if args.map:
        config.map = args.map
    if not args.food_decay is None:
        config.food_decay = args.food_decay or None
    if args.stream and args.nr_runs:
        raise ValueError('Cannot specify --nr-runs with --stream')
    if args.sound:
        config.sound = args.sound

def start_stream_run(conn, config):
    env = setup_env(config)
    env.stream_run(conn)


def main():
    argv = sys.argv[1:]
    global config
    ap = argparse.ArgumentParser()
    mutex = ap.add_mutually_exclusive_group(required=True)
    mutex.add_argument('--play-file', help='Play a saved run file')
    mutex.add_argument('--compute', action='store_true', help='Compute a run file')
    mutex.add_argument('--stream', action='store_true', help='compute and live-stream the run')
    ap.add_argument('--snake-count', type=int, help='Number of snakes to simulate')
    ap.add_argument('--grid-width', type=int, help='Width of the grid')
    ap.add_argument('--grid-height', type=int, help='Height of the grid')
    ap.add_argument('--food', type=int, help='Number of food to spawn')
    ap.add_argument('--food-decay', type=int, help='Number of steps before food decays, 0 for no decay')
    ap.add_argument('--calc-timeout', type=int, help='Timeout for calculation')
    ap.add_argument('--nr-runs', type=int, help='Number of runs to generate')
    ap.add_argument('--map', type=str, help='Path to map file')
    ap.add_argument('--sound', action='store_true', help='Play sound', default=False)
    args = ap.parse_args(argv)
    cfg_path = Path(__file__).parent / 'config/default_config.json'
    with open(cfg_path) as config_file:
        config = DotDict(json.load(config_file))
    handle_args(args, config)

    if args.play_file:
        play_runfile(filepath=Path(args.play_file), sound_on=config.sound)

    elif args.compute:
        env = setup_env(config)
        nr_runs = args.nr_runs or 1
        for _ in range(nr_runs):
            env.generate_run()
            env.reset()

    elif args.stream:
        parent_conn, child_conn = Pipe()
        env_p = Process(target=start_stream_run, args=(child_conn, config))
        render_p = Process(target=play_stream, args=(parent_conn, config.sound))
        render_p.start()
        env_p.start()
        render_p.join()
        parent_conn.send('stop')


if __name__ == '__main__':
   main()

