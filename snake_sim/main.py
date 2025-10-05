import sys
import json
import logging
import multiprocessing as mp
import ctypes
import cProfile

from pathlib import Path
from importlib import resources
from threading import Thread

from snake_sim.logging_setup import setup_logging
from snake_sim.environment.types import DotDict
from snake_sim.cli import cli
from snake_sim.environment.snake_loop_control import start_loop, setup_loop
from snake_sim.environment.interfaces.loop_observable_interface import ILoopObservable
from snake_sim.loop_observables.ipc_repeater_observable import IPCRepeaterObservable
from snake_sim.loop_observables.file_reader_observable import FileRepeaterObservable
from snake_sim.loop_observers.frame_builder_observer import FrameBuilderObserver
from snake_sim.loop_observers.state_builder_observer import StateBuilderObserver
from snake_sim.loop_observers.file_persist_observer import FilePersistObserver
from snake_sim.render.render_loop import RenderLoop, RenderConfig
from snake_sim.render.renderer_factory import renderer_factory

with resources.open_text('snake_sim.config', 'default_config.json') as config_file:
    default_config = DotDict(json.load(config_file))


log = logging.getLogger(Path(__file__).stem)


def main():
    try:
        argv = sys.argv[1:]
        cfg_path = Path(__file__).parent / 'config/default_config.json'
        with open(cfg_path) as config_file:
            config = DotDict(json.load(config_file))
        config = cli(argv, config)
        setup_logging(config.log_level)

        loop_repeater: ILoopObservable = None
        mp_ctx = mp.get_context("spawn")
        stop_flag = mp_ctx.Value(ctypes.c_bool, False)

        if config.command == "play-file":
            loop_repeater = FileRepeaterObservable(filepath=config.filepath)

        elif config.command == "compute":
            parent_conn, child_conn = mp_ctx.Pipe()
            loop_repeater = IPCRepeaterObservable(child_conn)
            if not config.no_record:
                store_dir = Path(config.record_dir) / f"grid_{config.grid_height}x{config.grid_width}"
                file_persist = FilePersistObserver(store_dir=store_dir, filename=config.record_file)
                loop_repeater.add_observer(file_persist)
            loop_p = mp_ctx.Process(target=start_loop, args=(config, stop_flag), kwargs={'ipc_observer_pipe': parent_conn})
            loop_p.start()
            Thread(target=loop_p.join).start()

        elif config.command == "game":
            raise NotImplementedError("Game mode not implemented yet")


        if not config.no_render:
            frame_builder = FrameBuilderObserver(config.expansion)
            state_builder = StateBuilderObserver()
            loop_repeater.add_observer(frame_builder)
            loop_repeater.add_observer(state_builder)
            render_config = RenderConfig(
                fps=config.fps,
                sound=False
            )
            renderer = renderer_factory(config.renderer, frame_builder)
            render_loop = RenderLoop(
                renderer=renderer,
                config=render_config,
                state_builder=state_builder,
                stop_flag=stop_flag
            )
            render_loop.start()

        loop_repeater.start()
        try:
            render_loop.join()
        except NameError:
            pass

    except KeyboardInterrupt:
        pass
    except Exception as e:
        log.error(e)
        log.debug("TRACE: ", exc_info=True)
    finally:
        try:
            stop_flag.value = True
        except:
            pass
        try:
            render_loop.stop()
        except:
            pass

if __name__ == '__main__':
    main()
    # cProfile.run('main()', sort='cumtime')