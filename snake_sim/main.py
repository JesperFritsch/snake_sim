import sys
import json
import logging
from multiprocessing import Pipe, Process, Event
from pathlib import Path
from importlib import resources
from typing import Dict, Tuple


from snake_sim.utils import DotDict
from snake_sim.render.pygame_render import play_runfile, play_stream, play_game
from snake_sim.cli import cli
from snake_sim.environment.snake_loop_control import SnakeLoopControl, SimConfig, GameConfig
from snake_sim.loop_observers.pygame_run_data_observer import PygameRunDataObserver
from snake_sim.loop_observers.recorder_run_data_observer import RecorderRunDataObserver
from snake_sim.loop_observers.run_data_loop_observer import RunDataLoopObserver
from snake_sim.data_adapters.run_data_adapter import RunDataAdapter

with resources.open_text('snake_sim.config', 'default_config.json') as config_file:
    default_config = DotDict(json.load(config_file))

log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
log = logging.getLogger()
log.setLevel(logging.DEBUG)
s_handler = logging.StreamHandler()
s_handler.formatter = logging.Formatter(log_format)
log.addHandler(s_handler)

def create_color_map(snake_ids) -> Dict[int, Tuple[int, int, int]]:
    color_map = {default_config[key]: value for key, value in default_config.color_mapping.items()}
    for i, id in enumerate(snake_ids):
        color_map[id] = default_config.snake_colors[i]["head_color"]
        color_map[id+1] = default_config.snake_colors[i]["body_color"]
    print(color_map)
    return color_map


def setup_sim_loop(config):
    loop_control = SnakeLoopControl()
    loop_control.init(SimConfig(
        map=config.map,
        food=config.food,
        height=config.grid_height,
        width=config.grid_width,
        food_decay=config.food_decay,
        snake_count=config.snake_count,
        calc_timeout=config.calc_timeout,
        verbose=config.verbose,
        start_length=config.start_length
    ))
    return loop_control


def setup_game_loop(config):
    loop_control = SnakeLoopControl()
    loop_control.init(GameConfig(
        map=config.map,
        food=config.food,
        height=config.grid_height,
        width=config.grid_width,
        food_decay=config.food_decay,
        snake_count=config.snake_count,
        calc_timeout=config.calc_timeout,
        verbose=config.verbose,
        player_count=config.num_players,
        spm=config.spm,
        start_length=config.start_length
    ))
    return loop_control


def main():

    argv = sys.argv[1:]
    cfg_path = Path(__file__).parent / 'config/default_config.json'
    with open(cfg_path) as config_file:
        config = DotDict(json.load(config_file))
    config = cli(argv, config)
    print(config)

    if config.command == "play-file":
        play_runfile(filepath=Path(config.filepath), sound_on=config.sound, fps=config.fps)

    elif config.command == "compute":
        for _ in range(config.nr_runs):
            loop_control = setup_sim_loop(config)
            run_data_loop_observer = RunDataLoopObserver()
            adapter = RunDataAdapter(loop_control.get_init_data(), create_color_map(loop_control.get_snake_ids()))
            run_data_loop_observer.set_adapter(adapter)
            if not config.no_record:
                recording_file = None if not config.record_file else config.record_file
                run_data_loop_observer.add_observer(RecorderRunDataObserver(recording_dir=config.record_dir, recording_file=recording_file, as_proto=False))
            loop_control.run()

    elif config.command == "stream":
        parent_conn, child_conn = Pipe()
        loop_control = setup_sim_loop(config)
        run_data_loop_observer = RunDataLoopObserver()
        adapter = RunDataAdapter(loop_control.get_init_data(), create_color_map(loop_control.get_snake_ids()))
        run_data_loop_observer.set_adapter(adapter)
        if not config.no_record:
            recording_file = None if not config.record_file else config.record_file
            run_data_loop_observer.add_observer(RecorderRunDataObserver(recording_dir=config.record_dir, recording_file=recording_file, as_proto=False))
        run_data_loop_observer.add_observer(PygameRunDataObserver(parent_conn))
        loop_control.add_observer(run_data_loop_observer)
        stop_event = Event()
        loop_p = Process(target=loop_control.run, args=(stop_event,))
        render_p = Process(target=play_stream, args=(child_conn, config.fps, config.sound))
        render_p.start()
        loop_p.start()
        render_p.join()
        stop_event.set()


    elif config.command == "game":
        parent_conn, child_conn = Pipe()
        loop_control = setup_game_loop(config)
        run_data_loop_observer = RunDataLoopObserver()
        adapter = RunDataAdapter(loop_control.get_init_data(), create_color_map(loop_control.get_snake_ids()))
        run_data_loop_observer.set_adapter(adapter)
        if not config.no_record:
            recording_file = None if not config.record_file else config.record_file
            run_data_loop_observer.add_observer(RecorderRunDataObserver(recording_dir=config.record_dir, recording_file=recording_file, as_proto=False))
        run_data_loop_observer.add_observer(PygameRunDataObserver(parent_conn))
        loop_control.add_observer(run_data_loop_observer)
        stop_event = Event()
        loop_p = Process(target=loop_control.run, args=(stop_event,))
        render_p = Process(target=play_stream, args=(child_conn, config.fps, config.sound))
        render_p.start()
        loop_p.start()
        render_p.join()
        stop_event.set()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        log.error(e)
        log.debug("TRACE: ", exc_info=True)

