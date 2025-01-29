import json
import logging
import sys
import functools

from typing import Union, List
from importlib import resources as pkg_resources

from snake_sim.environment.interfaces.main_loop_interface import ILoopObserver
from snake_sim.loop_observers.run_data_observer_interface import IRunDataObserver

from snake_sim.loop_observers.run_data_loop_observer import RunDataLoopObserver
from snake_sim.loop_observers.recorder_run_data_observer import RecorderRunDataObserver
from snake_sim.data_adapters.run_data_adapter import RunDataAdapter
from snake_sim.environment.main_loop import SimLoop, GameLoop
from snake_sim.environment.snake_handlers import SnakeHandler
from snake_sim.controllers.keyboard_controller import ControllerCollection
from snake_sim.environment.snake_factory import SnakeFactory
from snake_sim.environment.snake_env import SnakeEnv, EnvInitData
from snake_sim.environment.food_handlers import FoodHandler
from snake_sim.environment.snake_processes import SnakeProcessPool
from snake_sim.utils import get_map_files_mapping, DotDict, create_color_map
from dataclasses import dataclass

with pkg_resources.open_text('snake_sim.config', 'default_config.json') as config_file:
    default_config = DotDict(json.load(config_file))

log = logging.getLogger("main_loop")
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
handler.setFormatter(logging.Formatter("%(asctime)s:%(name)s:%(levelname)s:%(message)s"))
log.addHandler(handler)


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


@dataclass
class GameConfig(SimConfig):
    player_count: int
    spm: int


class SnakeLoopControl:
    def __init__(self, config: Union[SimConfig, GameConfig]):
        if not isinstance(config, (SimConfig, GameConfig)):
            raise ValueError('Invalid configuration')
        self._loop = None
        self._config = config
        self.process_pool = None
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

    def _loop_check(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if not self._loop:
                raise ValueError('Loop not initialized')
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
                snake.set_id(id)
                snake.set_start_length(self._config.start_length)
                snake.set_start_position(start_pos)
                snake.set_init_data(self._snake_enviroment.get_init_data())
            except Exception as e:
                log.exception(e)
                self._snake_handler.kill_snake(id)

    @_loop_check
    def _initialize_inproc_snakes(self):
        """ Initialize in-process snakes """
        snake_factory = SnakeFactory()
        auto_snakes = snake_factory.create_snakes({'auto': self._config.snake_count})
        for id, snake in auto_snakes.items():
            self._snake_handler.add_snake(id, snake)

    @_loop_check
    def _initialize_remotes(self):
        """ Initialize remote snakes """
        snake_factory = SnakeFactory()
        snake_processes = self.process_pool.get_running_processes()
        remote_snakes = [(snake_factory.get_next_id(), t) for t in self._config.external_snake_targets]
        remote_snakes.extend([(p.id, p.target) for p in snake_processes])
        for id, target in remote_snakes:
            id, snake = snake_factory.create_snake('remote', id=id, target=target)
            self._snake_handler.add_snake(id, snake)

    @_loop_check
    def _initialize_manual_snakes(self):
        """ Initialize manual snakes """
        snake_factory = SnakeFactory()
        ctl_collection = ControllerCollection()
        manual_snakes = snake_factory.create_snakes({'manual': self._config.player_count}, help=1)
        for id, snake in manual_snakes.items():
            self._snake_handler.add_snake(id, snake)
            ctl_collection.bind_controller(snake)
        ctl_collection.handle_controllers()

    @_loop_check
    def _spawn_snake_processes(self):
        """ Spawn the snake processes created internally """
        config = self._config
        self.process_pool = SnakeProcessPool()
        snake_factory = SnakeFactory()
        for _ in range(config.snake_count):
            self.process_pool.start(snake_factory.get_next_id())

    @_loop_check
    def add_observer(self, observer: ILoopObserver):
        """Adds an observer to the loop"""
        if not isinstance(observer, ILoopObserver):
            raise ValueError('Observer must be an instance of ILoopObserver')
        self._loop.add_observer(observer)

    @_loop_check
    def add_run_data_observer(self, observer: IRunDataObserver):
        """Adds a an observer to a RunDataLoopObserver"""
        if not isinstance(observer, IRunDataObserver):
            raise ValueError('Observer must be an instance of IRunDataObserver')

        observers = self._loop.get_observers()
        # if a RunDataLoopObserver already exists, add the observer to it
        try:
            run_data_observer = next(o for o in observers if isinstance(o, RunDataLoopObserver))
        except StopIteration:
            # if no RunDataLoopObserver exists, create one and add the observer to it
            run_data_observer = RunDataLoopObserver()
            self._loop.add_observer(run_data_observer)
        run_data_observer.add_observer(observer)

    @_loop_check
    def _initialize_run_data_loop_observers(self):
        """Initializes the run data loop observers
        This is used to initialize the DataAdapters for the RunDataLoopObservers
        It needs to happend after the snakes are added to the environment"""
        observers = self._loop.get_observers()
        for observer in observers:
            if isinstance(observer, RunDataLoopObserver):
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
    def run(self, stop_event=None):
        """Starts the loop
        Args:
            stop_event: Event object to stop the loop
        """
        self._spawn_snake_processes()
        self._initialize_remotes()
        # self._initialize_inproc_snakes()
        if isinstance(self._config, GameConfig):
            self._initialize_manual_snakes()
        self._finalize_snakes()
        self._initialize_run_data_loop_observers() # This needs to be called after the snakes are added to the environment
        try:
            self._loop.start(stop_event)
        except KeyboardInterrupt:
            log.info("Keyboard interrupt")
        except Exception as e:
            log.exception(e)
        finally:
            self.stop()

    @_loop_check
    def stop(self):
        """Stops the loop"""
        log.debug("Stopping loop")
        if self.process_pool:
            self.process_pool.shutdown()
        self._loop.stop()


def setup_loop(config) -> SnakeLoopControl:
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
            start_length=config.start_length,
            external_snake_targets=config.external_snake_targets
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
            start_length=config.start_length,
            external_snake_targets=config.external_snake_targets
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
    return loop_control