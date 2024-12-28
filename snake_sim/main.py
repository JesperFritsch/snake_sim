import sys
import json
import logging
from multiprocessing import Pipe, Process, Event
from pathlib import Path
from importlib import resources

from snake_sim.utils import DotDict
from snake_sim.render.pygame_render import play_runfile, play_stream, play_game
from snake_sim.cli import cli
from snake_sim.environment.snake_loop_control import setup_loop
from snake_sim.loop_observers.pygame_run_data_observer import PygameRunDataObserver

with resources.open_text('snake_sim.config', 'default_config.json') as config_file:
    default_config = DotDict(json.load(config_file))

log_format = "%(asctime)s:%(name)s:%(levelname)s:%(message)s"
log = logging.getLogger()
log.setLevel(logging.DEBUG)
s_handler = logging.StreamHandler()
s_handler.formatter = logging.Formatter(log_format)
log.addHandler(s_handler)


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
            loop_control = setup_loop(config)
            loop_control.run()

    elif config.command == "stream" or config.command == "game":
        parent_conn, child_conn = Pipe()
        loop_control = setup_loop(config)
        loop_control.add_run_data_observer(PygameRunDataObserver(parent_conn))
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

