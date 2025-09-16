import json
import logging
import atexit
import functools
import threading
from dataclasses import dataclass

from pathlib import Path
from typing import Union, List, Optional
from importlib import resources as pkg_resources

from snake_sim.data_adapters.run_data_adapter import RunDataAdapter

from snake_sim.loop_observers.tqdm_loop_observer import TqdmLoopObserver
from snake_sim.loop_observers.run_data_loop_source import RunDataSource
from snake_sim.loop_observers.recorder_run_data_observer import RecorderRunDataObserver
from snake_sim.environment.interfaces.main_loop_interface import ILoopObserver
from snake_sim.environment.interfaces.run_data_observer_interface import IRunDataObserver
from snake_sim.environment.main_loop import SimLoop, GameLoop
from snake_sim.environment.snake_handlers import SnakeHandler
from snake_sim.environment.snake_factory import SnakeFactory, SnakeProcType
from snake_sim.environment.snake_env import SnakeEnv, EnvInitData
from snake_sim.environment.food_handlers import FoodHandler
from snake_sim.environment.snake_processes import ProcessPool
from snake_sim.environment.types import DotDict, SnakeConfig

from snake_sim.utils import get_map_files_mapping, create_color_map

from snake_sim.snakes.strategies.utils import apply_strategies


with pkg_resources.open_text('snake_sim.config', 'default_config.json') as config_file:
    default_config = DotDict(json.load(config_file))


log = logging.getLogger(Path(__file__).stem)


@dataclass
class SimConfig:
    map: str
    food: int
    height: int
    width: int
    food_decay: int
    snake_count: int
    calc_timeout: int
    verbose: bool
    start_length: int
    external_snake_targets: List[str]
    inproc_snakes: bool
    snake_config: SnakeConfig


@dataclass
class GameConfig(SimConfig):
    player_count: int
    spm: int
    snake_game_config: SnakeConfig


class SnakeLoopControl:

    def __init__(self, config: Union[SimConfig, GameConfig]):
        if not isinstance(config, (SimConfig, GameConfig)):
            raise ValueError('Invalid configuration')
        self._loop = None
        self._config = config
        self.process_pool: ProcessPool = ProcessPool()
        self._snake_handler = SnakeHandler()
        self._snake_enviroment = SnakeEnv(
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
        self._snake_enviroment.set_food_handler(self._food_handler)
        if config.map:
            map_files_mapping = get_map_files_mapping()
            if config.map not in map_files_mapping:
                raise ValueError(f'Map {config.map} not found')
            self._snake_enviroment.load_map(map_files_mapping[config.map])
        self._is_shutdown = False
        atexit.register(self.shutdown)

    def _loop_check(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if not self._loop:
                raise ValueError('Loop not initialized, call init_loop() first')
            return func(self, *args, **kwargs)
        return wrapper

    def init_loop(self):
        """Initializes the loop"""
        if isinstance(self._config, GameConfig):
            # Initialize game loop add keyboard controllers
            self._loop = GameLoop()
            self._loop.set_steps_per_min(self._config.spm)

        else:
            # Initialize simulation loop
            self._loop = SimLoop()
        self._loop.set_snake_handler(self._snake_handler)
        self._loop.set_environment(self._snake_enviroment)
        if default_config.max_steps is not None:
            self._loop.set_max_steps(default_config.max_steps)
        if default_config.max_no_food_steps is not None:
            self._loop.set_max_no_food_steps(default_config.max_no_food_steps)
        else:
            self._loop.set_max_no_food_steps((self._snake_enviroment.get_init_data().height * self._snake_enviroment.get_init_data().width) // 2)

    @_loop_check
    def _finalize_snakes(self):
        """ Finalize snakes """
        for id, snake in self._snake_handler.get_snakes().items():
            start_pos = self._snake_enviroment.add_snake(id, start_length=self._config.start_length)
            try:
                log.debug(f"Snake {id} start position: {start_pos}")
                snake.set_id(id)
                snake.set_start_length(self._config.start_length)
                snake.set_start_position(start_pos)
                snake.set_init_data(self._snake_enviroment.get_init_data())
            except Exception as e:
                log.exception(e)
                self._snake_handler.kill_snake(id)
        self._snake_handler.finalize()

    @_loop_check
    def _initialize_inproc_snakes(self):
        """ Initialize in-process snakes """
        snake_factory = SnakeFactory()
        inproc_snakes = snake_factory.create_many_snakes(
            proc_type=SnakeProcType.LOCAL,
            snake_config=self._config.snake_config,
            count=self._config.snake_count - len(self._config.external_snake_targets)
        )
        remote_snake_targets = self._config.external_snake_targets.copy()
        for target in remote_snake_targets:
            id, snake = snake_factory.create_snake(SnakeProcType.REMOTE, target=target)
            self._snake_handler.add_snake(id, snake)

        for id, snake in inproc_snakes.items():
            self._snake_handler.add_snake(id, snake)
            apply_strategies(snake, self._config.snake_config)

    @_loop_check
    def _initialize_remotes(self):
        """ Initialize remote snakes """
        snake_factory = SnakeFactory()
        remote_snake_targets = self._config.external_snake_targets.copy()
        remote_snake_configs = [self._config.snake_config] * (self._config.snake_count - len(remote_snake_targets))

        for config in remote_snake_configs:
            id, snake = snake_factory.create_snake(SnakeProcType.REMOTE, snake_config=config)
            self._snake_handler.add_snake(id, snake)

        for target in remote_snake_targets:
            id, snake = snake_factory.create_snake(SnakeProcType.REMOTE, target=target)
            self._snake_handler.add_snake(id, snake)

    @_loop_check
    def add_observer(self, observer: ILoopObserver):
        """Adds an observer to the loop"""
        if not isinstance(observer, ILoopObserver):
            raise ValueError('Observer must be an instance of ILoopObserver')
        self._loop.add_observer(observer)

    @_loop_check
    def add_run_data_observer(self, observer: IRunDataObserver):
        """Adds a an observer to a RunDataSource"""
        if not isinstance(observer, IRunDataObserver):
            raise ValueError('Observer must be an instance of IRunDataObserver')

        observers = self._loop.get_observers()
        # if a RunDataSource already exists, add the observer to it
        try:
            run_data_observer = next(o for o in observers if isinstance(o, RunDataSource))
        except StopIteration:
            # if no RunDataSource exists, create one and add the observer to it
            run_data_observer = RunDataSource()
            self._loop.add_observer(run_data_observer)
        run_data_observer.add_observer(observer)

    @_loop_check
    def _initialize_run_data_loop_observers(self):
        """Initializes the run data loop observers
        This is used to initialize the DataAdapters for the RunDataSources
        It needs to happend after the snakes are added to the environment"""
        observers = self._loop.get_observers()
        for observer in observers:
            if isinstance(observer, RunDataSource):
                init_data = self._snake_enviroment.get_init_data()
                observer.set_adapter(
                    RunDataAdapter(
                        init_data,
                        create_color_map(init_data.snake_values)
                    )
                )

    @_loop_check
    def get_init_data(self) -> EnvInitData:
        """Returns the initial data of the environment"""
        return self._snake_enviroment.get_init_data()

    @_loop_check
    def run(self, stop_event: Optional[threading.Event] = None):
        """ Starts the loop """
        # If a stop event is provided, start a thread that waits for the event to be set
        if stop_event:
            def wait_stop_event(stop_event):
                try:
                    stop_event.wait()
                except (ConnectionResetError, BrokenPipeError):
                    # Manager is already dead, just exit gracefully
                    pass
                self.shutdown()
            threading.Thread(target=wait_stop_event, args=(stop_event,), daemon=True).start()

        if self._config.inproc_snakes:
            self._initialize_inproc_snakes()
        else:
            self._initialize_remotes()
        self._finalize_snakes()
        self._initialize_run_data_loop_observers() # This needs to be called after the snakes are added to the environment
        try:
            self._loop.start()
        except KeyboardInterrupt:
            pass
        except Exception as e:
            log.exception(e)
        finally:
            self.shutdown()

    @_loop_check
    def shutdown(self):
        if self._is_shutdown:
            return
        """Shuts down the loop"""
        self._is_shutdown = True
        self._loop.stop()
        self.process_pool.shutdown()


def setup_loop(config) -> SnakeLoopControl:
    sim_config = SimConfig(
        map=config.map,
        food=config.food,
        height=config.grid_height,
        width=config.grid_width,
        food_decay=config.food_decay,
        snake_count=config.snake_count,
        calc_timeout=config.calc_timeout,
        verbose=config.verbose,
        start_length=config.start_length,
        external_snake_targets=config.external_snake_targets,
        inproc_snakes=config.inproc_snakes,
        snake_config=config.snake_config,
    )
    if config.command == "game":
        sim_config = GameConfig(
            **sim_config.__dict__,
            player_count=config.num_players,
            spm=config.spm,
        )
    loop_control = SnakeLoopControl(sim_config)
    loop_control.init_loop()
    if not config.no_record:
        recording_file = config.record_file if config.record_file else None
        loop_control.add_run_data_observer(
            RecorderRunDataObserver(
                recording_dir=config.record_dir,
                recording_file=recording_file,
            )
        )
    if config.rate_meter:
        loop_control.add_observer(TqdmLoopObserver())

    return loop_control
