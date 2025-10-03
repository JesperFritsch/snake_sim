import sys
import json
import logging
import multiprocessing as mp
import ctypes
from multiprocessing.sharedctypes import Synchronized
from pathlib import Path
from importlib import resources

from snake_sim.logging_setup import setup_logging
from snake_sim.environment.types import DotDict
from snake_sim.render.pygame_render_old import play_runfile, play_stream, play_game
from snake_sim.cli import cli
from snake_sim.environment.snake_loop_control import setup_loop
from snake_sim.loop_observers.ipc_repeater_observer import IPCRepeaterObserver
from snake_sim.loop_observables.ipc_repeater_observable import IPCRepeaterObservable
from snake_sim.loop_observers.frame_builder_observer import FrameBuilderObserver
from snake_sim.loop_observers.state_builder_observer import StateBuilderObserver
from snake_sim.render.render_loop import RenderLoop, RenderConfig
from snake_sim.render.terminal_render import TerminalRenderer
from snake_sim.render.pygame_render import PygameRenderer


with resources.open_text('snake_sim.config', 'default_config.json') as config_file:
    default_config = DotDict(json.load(config_file))


log = logging.getLogger(Path(__file__).stem)


def start_snakes(config: DotDict, stop_flag: Synchronized, ipc_observer_pipe=None):
    # in linux the process is forked and inherits the loggers, but on windows we need to set it up again
    if not logging.getLogger().hasHandlers():
        setup_logging(config.log_level)
    loop_control = setup_loop(config)
    if ipc_observer_pipe:
        # loop_control.add_run_data_observer(IPCRunDataObserver(ipc_observer_pipe))
        loop_control.add_observer(IPCRepeaterObserver(ipc_observer_pipe))
    loop_control.run(stop_flag)
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
            mp_ctx = mp.get_context("spawn")
            parent_conn, child_conn = mp_ctx.Pipe()
            stop_flag = mp_ctx.Value(ctypes.c_bool, False)
            loop_p = mp_ctx.Process(target=start_snakes, args=(config, stop_flag), kwargs={'ipc_observer_pipe': parent_conn})

            frame_builder = FrameBuilderObserver(2)
            state_builder = StateBuilderObserver()

            loop_repeater = IPCRepeaterObservable(child_conn)
            loop_repeater.add_observer(frame_builder)
            loop_repeater.add_observer(state_builder)

            loop_p.start()

            render_config = RenderConfig(
                fps=config.fps,
                sound=False
            )
            # renderer = TerminalRenderer(frame_builder)
            renderer = PygameRenderer(frame_builder)
            
            render_loop = RenderLoop(
                renderer=renderer,
                config=render_config,
                state_builder=state_builder,
                stop_flag=stop_flag
            )
            # render_p.start()
            render_loop.start()
            # render_p.join()
            loop_p.join()

    except KeyboardInterrupt:
        pass
    except Exception as e:
        log.error(e)
        log.debug("TRACE: ", exc_info=True)
    finally:
        stop_flag.value = True
        try:
            render_loop.stop()
        except:
            pass

if __name__ == '__main__':
    main()
