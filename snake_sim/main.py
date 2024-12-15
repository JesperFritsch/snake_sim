import sys
import json
from multiprocessing import Pipe, Process
from pathlib import Path

from snake_sim.utils import DotDict
from snake_sim.render.pygame_render import play_runfile, play_stream, play_game
from snake_sim.cli import cli


from snake_sim.environment.snake_loop_control import SnakeLoopControl, SimConfig, GameConfig
from snake_sim.loop_observers.pygame_run_data_observer import PygameRunDataObserver
from snake_sim.loop_observers.recorder_run_data_observer import RecorderRunDataObserver
from snake_sim.loop_observers.run_data_loop_observer import RunDataLoopObserver

def setup_sim_env(config):
    env = SnakeLoopControl()
    env.init(SimConfig(
        map=config.map,
        food=config.food,
        height=config.grid_height,
        width=config.grid_width,
        food_decay=config.food_decay,
        snake_count=config.snake_count,
        calc_timeout=config.calc_timeout,
        verbose=config.verbose
    ))
    return env


def setup_game_env(config):
    env = SnakeLoopControl()
    env.init(GameConfig(
        map=config.map,
        food=config.food,
        height=config.grid_height,
        width=config.grid_width,
        food_decay=config.food_decay,
        snake_count=config.snake_count,
        calc_timeout=config.calc_timeout,
        verbose=config.verbose,
        player_count=config.player_count,
        spm=config.spm
    ))
    return env


def main():

    argv = sys.argv[1:]
    cfg_path = Path(__file__).parent / 'config/default_config.json'
    with open(cfg_path) as config_file:
        config = DotDict(json.load(config_file))
    config = cli(argv, config)

    if config.command == "play-file":
        play_runfile(filepath=Path(config.filepath), sound_on=config.sound, fps=config.fps)

    elif config.command == "compute":
        for _ in range(config.nr_runs):
            env = setup_sim_env(config)
            env.run()

    elif config.command == "stream":
        parent_conn, child_conn = Pipe()
        env = setup_sim_env(config)
        loop_observer = RunDataLoopObserver()
        recording_file = None if not config.record_file else config.record_file
        loop_observer.add_observer(RecorderRunDataObserver(recording_file, False))
        loop_observer.add_observer(PygameRunDataObserver(parent_conn))
        env.add_observer(loop_observer)
        env_p = Process(target=env.run)
        render_p = Process(target=play_stream, args=(child_conn, config.fps, config.sound))
        render_p.start()
        env_p.start()
        render_p.join()


    # elif config.command == "game":
    #     parent_conn, child_conn = Pipe()
    #     env_p = Process(target=start_game_run, args=(child_conn, config))
    #     # since the FrameBuilder by default expands the frame by 2, each step is 2 frames,
    #     # but this should not be hardcoded like this, but figure it out later...
    #     fps = (config.spm * 2) / 60
    #     render_p = Process(target=play_game, args=(parent_conn, fps, config.sound))
    #     render_p.start()
    #     env_p.start()
    #     render_p.join()
    #     parent_conn.send('stop')


if __name__ == '__main__':
    main()

