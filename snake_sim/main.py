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
from snake_sim.loop_observers.ipc_run_data_observer import IPCRunDataObserver


with resources.open_text('snake_sim.config', 'default_config.json') as config_file:
    default_config = DotDict(json.load(config_file))

log = logging.getLogger()

def setup_logging(log_level):
    handler = logging.StreamHandler()
    handler.setLevel(log_level)
    handler.setFormatter(logging.Formatter("%(asctime)s:%(name)s:%(levelname)s:%(message)s"))
    log.addHandler(handler)


def main():
    try:
        argv = sys.argv[1:]
        cfg_path = Path(__file__).parent / 'config/default_config.json'
        with open(cfg_path) as config_file:
            config = DotDict(json.load(config_file))
        config = cli(argv, config)
        setup_logging(config.log_level)

        if config.command == "play-file":
            play_runfile(filepath=Path(config.filepath), sound_on=config.sound, fps=config.fps)

        elif config.command == "compute":
            for _ in range(config.nr_runs):
                loop_control = setup_loop(config)
                loop_control.run()

        elif config.command == "stream" or config.command == "game":
            parent_conn, child_conn = Pipe()
            loop_control = setup_loop(config)
            loop_control.add_run_data_observer(IPCRunDataObserver(parent_conn))
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

    except KeyboardInterrupt:
        log.info("Keyboard interrupt")
    except Exception as e:
        log.error(e)
        log.debug("TRACE: ", exc_info=True)

if __name__ == '__main__':
    main()

