import os
import sys
import datetime
import json
import argparse
from multiprocessing import Pipe, Process
from pathlib import Path

sys.path.append(os.getcwd())


from utils import DotDict
from snake_env import SnakeEnv
from snakes.autoSnake import AutoSnake
from snakes.autoSnake2 import AutoSnake2
from snakes.autoSnake3 import AutoSnake3
from snakes.autoSnake4 import AutoSnake4

def setup_env(config):
    env = SnakeEnv(config.GRID_WIDTH, config.GRID_HEIGHT, config.FOOD)
    count = 0
    for snake_config in config.snake_configs:
        count += 1
        env.add_snake(AutoSnake4(**snake_config['snake']), **snake_config['env'])
        if count == config.snake_count:
            break
    return env


def handle_args(args):
    global config

    if args.grid_width:
        config.GRID_WIDTH = args.grid_width
    if args.grid_height:
        config.GRID_HEIGHT = args.grid_height
    if args.food:
        config.FOOD = args.food
    if args.snake_count:
        config.snake_count = args.snake_count

    if args.stream:
        if args.nr_runs:
            raise ValueError('Cannot specify --nr-runs with --stream')

def start_stream_run(conn, config):
    env = setup_env(config)
    env.stream_run(conn,)


def main(argv):
    ap = argparse.ArgumentParser()
    mutex = ap.add_mutually_exclusive_group(required=True)
    mutex.add_argument('--play-file', help='Play a saved run file')
    mutex.add_argument('--compute', action='store_true', help='Compute a run file')
    mutex.add_argument('--stream', action='store_true', help='compute and live-stream the run')
    ap.add_argument('--snake_count', type=int, help='Number of snakes to simulate')
    ap.add_argument('--grid_width', type=int, help='Width of the grid')
    ap.add_argument('--grid_height', type=int, help='Height of the grid')
    ap.add_argument('--food', type=int, help='Number of food to spawn')
    ap.add_argument('--nr-runs', type=int, help='Number of runs to generate')
    args = ap.parse_args(argv)

    with open('default_config.json') as config_file:
        config = DotDict(json.load(config_file))
    print(config.snake_count)
    handle_args(args)

    if args.play_file:
        from render.pygame_render import play_runfile
        play_runfile(Path(args.play_file))

    elif args.compute:
        env = setup_env(config)
        nr_runs = args.nr_runs or 1
        for _ in range(nr_runs):
            env.generate_run()
            env.reset()

    elif args.stream:
        from render.pygame_render import play_stream
        parent_conn, child_conn = Pipe()
        env_p = Process(target=start_stream_run, args=(child_conn, config))
        render_p = Process(target=play_stream, args=(parent_conn,))
        render_p.start()
        env_p.start()
        render_p.join()
        parent_conn.send('stop')


if __name__ == '__main__':
   main(sys.argv[1:])

