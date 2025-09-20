import sys
import json
import logging
from multiprocessing import Pipe, Process, Event as p_Event
from multiprocessing.synchronize import Event as MPEvent
from pathlib import Path
from importlib import resources

from snake_sim.logging_setup import setup_logging
from snake_sim.environment.types import DotDict
from snake_sim.render.pygame_render import play_runfile, play_stream, play_game
from snake_sim.cli import cli
from snake_sim.environment.snake_loop_control import setup_loop
from snake_sim.loop_observers.ipc_run_data_observer import IPCRunDataObserver


with resources.open_text('snake_sim.config', 'default_config.json') as config_file:
    default_config = DotDict(json.load(config_file))


log = logging.getLogger(Path(__file__).stem)


def start_snakes(config: DotDict, stop_event: MPEvent, ipc_observer_pipe=None):
    # in linux the process is forked and inherits the loggers, but on windows we need to set it up again
    if not logging.getLogger().hasHandlers():
        setup_logging(config.log_level)
    loop_control = setup_loop(config)
    if ipc_observer_pipe:
        loop_control.add_run_data_observer(IPCRunDataObserver(ipc_observer_pipe))
    loop_control.run(stop_event)
    sys.stdout.flush()


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
            stop_event = p_Event()
            loop_p = Process(target=start_snakes, args=(config, stop_event), kwargs={'ipc_observer_pipe': parent_conn})
            if config.command == "game":
                render_p = Process(target=play_game, args=(child_conn, config.spm, config.sound, stop_event))
            else:
                render_p = Process(target=play_stream, args=(child_conn, config.fps, config.sound, stop_event))
            render_p.start()
            loop_p.start()
            render_p.join()
            # stop_event.set()
            loop_p.join()

    except KeyboardInterrupt:
        pass
    except Exception as e:
        log.error(e)
        log.debug("TRACE: ", exc_info=True)
    finally:
        try:
            stop_event.set()
        except:
            pass

if __name__ == '__main__':
    main()
