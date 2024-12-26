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

log_format = "%(asctime)s:%(name)s:%(levelname)s:%(message)s"
log = logging.getLogger()
log.setLevel(logging.DEBUG)
s_handler = logging.StreamHandler()
s_handler.formatter = logging.Formatter(log_format)
log.addHandler(s_handler)


def create_color_map(env_init_data) -> Dict[int, Tuple[int, int, int]]:
    snake_values = env_init_data.snake_values
    color_map = {default_config[key]: value for key, value in default_config.color_mapping.items()}
    for i, snake_value_dict in enumerate(snake_values.values()):
        color_map[snake_value_dict["head_value"]] = default_config.snake_colors[i]["head_color"]
        color_map[snake_value_dict["body_value"]] = default_config.snake_colors[i]["body_color"]
    return color_map

def setup_loop(config, run_data_loop_observer: RunDataLoopObserver):
    loop_control = SnakeLoopControl()
    if config.command == "stream" or config.command == "compute":
        sim_config = SimConfig(
            map=config.map,
            food=config.food,
            height=config.grid_height,
            width=config.grid_width,
            food_decay=config.food_decay,
            snake_count=config.snake_count,
            calc_timeout=config.calc_timeout,
            verbose=config.verbose,
            start_length=config.start_length
        )
    elif config.command == "game":
        sim_config = GameConfig(
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
        )
    loop_control.init(sim_config)
    init_data = loop_control.get_init_data()
    adapter = RunDataAdapter(init_data, create_color_map(init_data))
    run_data_loop_observer.set_adapter(adapter)
    if not config.no_record:
        recording_file = config.record_file if config.record_file else None
        run_data_loop_observer.add_observer(RecorderRunDataObserver(recording_dir=config.record_dir, recording_file=recording_file, as_proto=True))
    loop_control.add_observer(run_data_loop_observer)
    return loop_control


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
            run_data_loop_observer = RunDataLoopObserver()
            loop_control = setup_loop(config, run_data_loop_observer)
            loop_control.run()

    elif config.command == "stream" or config.command == "game":
        parent_conn, child_conn = Pipe()
        run_data_loop_observer = RunDataLoopObserver()
        loop_control = setup_loop(config, run_data_loop_observer)
        run_data_loop_observer.add_observer(PygameRunDataObserver(parent_conn))
        stop_event = Event()
        loop_p = Process(target=loop_control.run, args=(stop_event,))
        if config.command == "game":
            render_p = Process(target=play_game, args=(child_conn, config.spm, config.sound))
        else:
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

