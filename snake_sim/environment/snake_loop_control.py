import json
import logging
import sys
import time
import functools
import threading
from dataclasses import dataclass

from pathlib import Path
from typing import Union, List, Optional
from importlib import resources as pkg_resources
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as concurrentTimeoutError
from multiprocessing.sharedctypes import Synchronized

from snake_sim.loop_observers.ipc_repeater_observer import IPCRepeaterObserver
from snake_sim.loop_observers.tqdm_loop_observer import TqdmLoopObserver
from snake_sim.environment.interfaces.main_loop_interface import ILoopObserver
from snake_sim.loop_observables.main_loop import SimLoop, GameLoop
from snake_sim.environment.snake_handlers import SnakeHandler
from snake_sim.environment.snake_factory import SnakeFactory
from snake_sim.environment.snake_env import SnakeEnv, EnvMetaData
from snake_sim.environment.food_handlers import FoodHandler
from snake_sim.environment.snake_processes import SnakeProcessManager
from snake_sim.environment.types import DotDict, SnakeConfig, SnakeProcType, SimConfig, GameConfig, StrategyConfig
from snake_sim.logging_setup import setup_logging
from snake_sim.map_utils.general import get_map_files_mapping

with pkg_resources.open_text('snake_sim.config', 'default_config.json') as config_file:
    default_config = DotDict(json.load(config_file))


log = logging.getLogger(Path(__file__).stem)


class SnakeLoopControl:

    def __init__(self, config: Union[SimConfig, GameConfig]):
        if not isinstance(config, (SimConfig, GameConfig)):
            raise ValueError('Invalid configuration')
        self._loop = None
        self._config = config
        self._snake_proc_mngr: SnakeProcessManager = SnakeProcessManager()
        self._snake_handler = SnakeHandler()
        self._snake_environment = SnakeEnv(
            config.width,
            config.height,
            default_config.free_value,
            default_config.blocked_value,
            default_config.food_value)
        self._food_handler = FoodHandler(
            config.width,
            config.height,
            config.food,
            config.food_decay)
        self._snake_environment.set_food_handler(self._food_handler)
        if config.map:
            map_files_mapping = get_map_files_mapping()
            if config.map in map_files_mapping:
                map_path = map_files_mapping[config.map]
            else:
                log.debug(f"Map with name '{config.map}' not found")
                map_path = config.map
            self._snake_environment.load_map(map_path)
        self._is_shutdown = False

    def _loop_check(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if not self._loop:
                raise ValueError('Loop not initialized, call init_loop() first')
            return func(self, *args, **kwargs)
        return wrapper

    def init_loop(self):
        """Initializes the loop"""
        try:
            if isinstance(self._config, GameConfig):
                # Initialize game loop add keyboard controllers
                self._loop = GameLoop()
                self._loop.set_steps_per_sec(self._config.steps_per_sec)
                self._initialize_player_snakes()
            else:
                # Initialize simulation loop
                self._loop = SimLoop()

            if self._config.distributed_snakes:
                self._initialize_distributed_snakes()
            else:
                self._initialize_inproc_snakes()

            self._initialize_remote_grpcs()
            self._finalize_snakes()
            
            self._loop.set_snake_handler(self._snake_handler)
            self._loop.set_environment(self._snake_environment)
            if default_config.max_steps is not None:
                self._loop.set_max_steps(default_config.max_steps)
            self._loop.set_decision_timout(self._config.decision_timeout)
            if default_config.max_no_food_steps is not None:
                self._loop.set_max_no_food_steps(default_config.max_no_food_steps)
            else:
                self._loop.set_max_no_food_steps((self._snake_environment.get_init_data().height * self._snake_environment.get_init_data().width) // 2)
            self._loop.set_end_on_last_standing_when_longest(
                self._config.end_on_last_standing_when_longest
            )
            if self._config.end_when_dead_tag is not None:
                self._loop.set_end_when_dead_tag(
                    self._config.end_when_dead_tag,
                    self._config.end_when_dead_buffer_steps,
                )
        except:
            self.shutdown()
            raise

    @_loop_check
    def _finalize_snakes(self):
        """ Finalize snakes """
        snakes_dict = self._snake_handler.get_snakes().copy()
        for id, snake in snakes_dict.items():
            tag = self._snake_handler.get_snake_tag(id)
            start_pos = self._snake_environment.add_snake(id, tag, start_length=self._config.start_length)
            try:
                log.debug(f"Snake {id} ({tag}) start position: {start_pos}")
                snake.set_id(id)
                snake.set_start_length(self._config.start_length)
                snake.set_start_position(start_pos)
            except Exception as e:
                log.exception(e)
                self._snake_handler.kill_snake(id)

        # Now that all snakes are added to the environment, we can finalize them with the init data
        init_data = self._snake_environment.get_init_data()
        for id, snake in snakes_dict.items():
            snake.set_init_data(init_data)
        self._snake_handler.finalize(init_data)

    @_loop_check
    def _initialize_inproc_snakes(self):
        """ Initialize in-process snakes """
        snake_factory = SnakeFactory()
        for i, snake_config in enumerate(self._config.snake_configs):
            snake = snake_factory.create_snake(snake_config=snake_config)
            tag = f"{snake.__class__.__name__}_{i}"
            self._snake_handler.add_snake(snake, tag)

    def _initialize_single_remote_snake(self, snake_factory: SnakeFactory, s_config: SnakeConfig):
        try:
            snake = snake_factory.create_snake(snake_config=s_config)
            self._snake_handler.add_snake(snake, s_config.tag)
        except Exception as e:
            log.exception(e)

    @_loop_check
    def _initialize_remote_grpcs(self):
        """ Initialize remote snakes """
        if not self._config.external_snake_configs:
            return
        threadpool = ThreadPoolExecutor(max_workers=len(self._config.external_snake_configs))
        snake_factory = SnakeFactory()
        futures = []
        for s_config in self._config.external_snake_configs:
            futures.append(threadpool.submit(self._initialize_single_remote_snake, snake_factory, s_config))
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                log.exception(e)

    @_loop_check
    def _initialize_distributed_snakes(self):
        snake_factory = SnakeFactory()
        non_manuals = [sc for sc in self._config.snake_configs if not any(c.type == "manual" for c in sc.strategies.values())]
        for snake_config in non_manuals:
            snake_id = self._snake_handler.get_next_snake_id()
            self._snake_proc_mngr.start(
                id=snake_id,
                proc_type=SnakeProcType.GRPC,
                snake_config=snake_config
            )
            target = self._snake_proc_mngr.get_target(snake_id)
            proxy_config = SnakeConfig(
                type="remote",
                tag=target,
                args=DotDict(
                    target=target,
                    timeout=self._config.ext_conn_timeout,
                    init_timeout=self._config.ext_init_timeout
                )
            )
            self._initialize_single_remote_snake(snake_factory, proxy_config)

    @_loop_check
    def _initialize_player_snakes(self):
        snake_factory = SnakeFactory()
        for i, snake_config in enumerate(self._config.player_snake_congfigs):
            snake = snake_factory.create_snake(snake_config=snake_config)
            tag = f"Manual_{i}"
            self._snake_handler.add_snake(snake, tag)

    @_loop_check
    def add_observer(self, observer: ILoopObserver):
        """Adds an observer to the loop"""
        if not isinstance(observer, ILoopObserver):
            raise ValueError('Observer must be an instance of ILoopObserver')
        self._loop.add_observer(observer)

    @_loop_check
    def get_init_data(self) -> EnvMetaData:
        """Returns the initial data of the environment"""
        return self._snake_environment.get_init_data()

    @_loop_check
    def run(self, stop_flag: Optional[Synchronized] = None):
        """ Starts the loop """
        try:
            if stop_flag is not None:
                def wait_stop_flag(stop_flag: Synchronized):
                    while not stop_flag.value:
                        time.sleep(0.01)
                    log.debug("Stop flag set, stopping loop")
                    self._loop.stop()
                threading.Thread(target=wait_stop_flag, args=(stop_flag,), daemon=True).start()
            self._loop.start()
        except KeyboardInterrupt:
            log.debug("KeyboardInterrupt received, shutting down loop")
        except Exception as e:
            log.exception(e)
        finally:
            self.shutdown()

    @_loop_check
    def shutdown(self):
        """Shuts down the loop"""
        if self._is_shutdown:
            return
        log.debug("Shutting down loop")
        self._is_shutdown = True
        self._loop.stop()
        self._loop.close()
        self._snake_handler.close()
        self._snake_proc_mngr.shutdown()
        log.debug("Loop shutdown complete")


def setup_loop(config) -> SnakeLoopControl:
    sim_config = SimConfig(
        map=config.map,
        food=config.food,
        height=config.grid_height,
        width=config.grid_width,
        food_decay=config.food_decay,
        snake_count=config.snake_count,
        calc_timeout=config.calc_timeout,
        start_length=config.start_length,
        distributed_snakes=config.distributed_snakes,
        ext_conn_timeout=config.ext_conn_timeout,
        ext_init_timeout=config.ext_init_timeout,
        external_snake_configs=[
            SnakeConfig(
                "remote",
                target,
                DotDict(
                    target=target,
                    timeout=config.ext_conn_timeout,
                    init_timeout=config.ext_init_timeout
                )
            )
            for target in config.ext_targets
        ],
        snake_configs=[
            SnakeConfig.from_dict({**config[config.snake_config_key], "tag": f"Snake_{i}"} if config.snake_config_key else {**config.snake_config, "tag": f"Snake_{i}"})
            for i in range(config.snake_count)
        ],
        decision_timeout=config.decision_timeout_ms if config.decision_timeout_ms > 0 else None,
        end_on_last_standing_when_longest=bool(config.end_on_last_standing_when_longest),
        end_when_dead_tag=config.end_when_dead_tag,
        end_when_dead_buffer_steps=int(config.end_when_dead_buffer_steps or 0),
    )
    if config.command == "game":
        sim_config = GameConfig(
            **sim_config.__dict__,
            player_count=config.num_players,
            steps_per_sec=config.game_sps,
            player_snake_congfigs=config.player_snake_configs
        )
    loop_control = SnakeLoopControl(sim_config)
    loop_control.init_loop()
    if config.rate_meter:
        loop_control.add_observer(TqdmLoopObserver())

    return loop_control

def start_loop(config: DotDict, stop_flag: Synchronized, ipc_observer_pipe=None):
    try:
        if not logging.getLogger().hasHandlers():
            setup_logging(config.log_level, config.log_dir)
        loop_control = setup_loop(config)
        if ipc_observer_pipe:
            loop_control.add_observer(IPCRepeaterObserver(ipc_observer_pipe))
        loop_control.run(stop_flag)
        sys.stdout.flush()
    except Exception as e:
        log.exception(e)
        try:
            loop_control.shutdown()
        except:
            pass
