import sys
import json
from multiprocessing import Pipe, Process
from pathlib import Path

from snake_sim.controllers.keyboard_controller import ControllerCollection
from snake_sim.utils import DotDict
from snake_sim.snake_env import SnakeEnv
from snake_sim.snakes.auto_snake import AutoSnake
from snake_sim.snakes.manual_snake import ManualSnake
from snake_sim.render.pygame_render import play_runfile, play_stream, play_game
from snake_sim.cli import cli

def setup_game(config):
    env = SnakeEnv(config.grid_width, config.grid_height, config.food, config.food_decay)
    if config.get('map'):
        env.load_png_map(config.map)
    count = config.num_players

    ctl_collection = ControllerCollection()
    for player in range(config.num_players):
        snake_config = config.snake_configs[player]
        man_snake = ManualSnake(**snake_config['snake'], help=1)
        ctl_collection.bind_controller(man_snake)
        env.add_snake(man_snake, **snake_config['env'])
    ctl_collection.handle_controllers() # this reads the keyboard input in a separate thread

    for snake_config in config.snake_configs[config.num_players:]:
        if count >= config.snake_count:
            break
        count += 1
        env.add_snake(AutoSnake(**snake_config['snake'], calc_timeout=config.calc_timeout), **snake_config['env'])
    return env


def setup_env(config):
    env = SnakeEnv(config.grid_width, config.grid_height, config.food, config.food_decay)
    if config.get('map'):
        env.load_png_map(config.map)
    count = 0

    for snake_config in config.snake_configs:
        if count >= config.snake_count:
            break
        count += 1
        env.add_snake(AutoSnake(**snake_config['snake'], calc_timeout=config.calc_timeout), **snake_config['env'])
    return env


def start_game_run(conn, config):
    env = setup_game(config)
    env.game_run(conn, steps_per_min=config.spm, verbose=config.verbose)


def start_stream_run(conn, config):
    env = setup_env(config)
    env.stream_run(conn, verbose=config.verbose)


def main():

    argv = sys.argv[1:]
    cfg_path = Path(__file__).parent / 'config/default_config.json'
    with open(cfg_path) as config_file:
        config = DotDict(json.load(config_file))
    config = cli(argv, config)

    if config.command == "play-file":
        play_runfile(filepath=Path(config.filepath), sound_on=config.sound, fps=config.fps)

    elif config.command == "compute":
        env = setup_env(config)
        nr_runs = config.nr_runs or 1
        for _ in range(nr_runs):
            env.generate_run(verbose=config.verbose)
            env.reset()

    elif config.command == "stream":
        parent_conn, child_conn = Pipe()
        env_p = Process(target=start_stream_run, args=(child_conn, config))
        render_p = Process(target=play_stream, args=(parent_conn, config.fps, config.sound))
        render_p.start()
        env_p.start()
        render_p.join()
        parent_conn.send('stop')

    elif config.command == "game":
        parent_conn, child_conn = Pipe()
        env_p = Process(target=start_game_run, args=(child_conn, config))
        # since the FrameBuilder by default expands the frame by 2, each step is 2 frames,
        # but this should not be hardcoded like this, but figure it out later...
        fps = (config.spm * 2) / 60
        render_p = Process(target=play_game, args=(parent_conn, fps, config.sound))
        render_p.start()
        env_p.start()
        render_p.join()

if __name__ == '__main__':
    main()

