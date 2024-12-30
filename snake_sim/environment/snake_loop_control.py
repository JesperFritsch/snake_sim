import json
import logging
import functools
from typing import Union
from importlib import resources as pkg_resources

from snake_sim.environment.interfaces.main_loop_interface import ILoopObserver
from snake_sim.loop_observers.run_data_observer_interface import IRunDataObserver

from snake_sim.loop_observers.run_data_loop_observer import RunDataLoopObserver
from snake_sim.loop_observers.recorder_run_data_observer import RecorderRunDataObserver
from snake_sim.data_adapters.run_data_adapter import RunDataAdapter
from snake_sim.snakes.manual_snake import ManualSnake
from snake_sim.snakes.remote_snake import RemoteSnake
from snake_sim.environment.main_loop import SimLoop, GameLoop
from snake_sim.environment.snake_handlers import SnakeHandler
from snake_sim.controllers.keyboard_controller import ControllerCollection
from snake_sim.environment.snake_factory import SnakeFactory
from snake_sim.environment.snake_env import SnakeEnv, EnvInitData
from snake_sim.environment.food_handlers import FoodHandler
from snake_sim.utils import get_map_files_mapping, DotDict, create_color_map
from dataclasses import dataclass

with pkg_resources.open_text('snake_sim.config', 'default_config.json') as config_file:
    default_config = DotDict(json.load(config_file))

log = logging.getLogger("main_loop")
log.setLevel(logging.DEBUG)

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


@dataclass
class GameConfig(SimConfig):
    player_count: int
    spm: int


class SnakeLoopControl:
    def __init__(self):
        self._loop = None

    def init_check(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if not self._loop:
                raise ValueError('Loop not initialized')
            return func(self, *args, **kwargs)
        return wrapper

    def init(self, config: Union[SimConfig, GameConfig]):
        """Initializes the environment with the given configuration"""
        if not isinstance(config, (SimConfig, GameConfig)):
            raise ValueError('Invalid configuration')
        self._snake_handler = SnakeHandler()
        snake_factory = SnakeFactory()
        self._snake_enviroment = SnakeEnv(config.width, config.height, default_config.free_value, default_config.blocked_value, default_config.food_value)
        self._food_handler = FoodHandler(config.width, config.height, config.food, config.food_decay)
        self._snake_enviroment.set_food_handler(self._food_handler)

        if config.map:
            map_files_mapping = get_map_files_mapping()
            if config.map not in map_files_mapping:
                raise ValueError(f'Map {config.map} not found')
            self._snake_enviroment.load_map(map_files_mapping[config.map])

        if isinstance(config, GameConfig):
            # Initialize game loop add keyboard controllers
            self._loop = GameLoop()
            self._loop.set_steps_per_min(config.spm)
            for man_snake in snake_factory.create_snakes({'manual': config.player_count}, start_length=config.start_length, help=1):
                self._snake_handler.add_snake(man_snake)

            for auto_snake in snake_factory.create_snakes({'auto': config.snake_count - config.player_count}, start_length=config.start_length):
                self._snake_handler.add_snake(auto_snake)

        else:
            # Initialize simulation loop
            self._loop = SimLoop()
            remote_snake = snake_factory.create_snake('remote', id=config.snake_count, start_length=config.start_length, target='localhost:50051')
            self._snake_handler.add_snake(remote_snake)
            for auto_snake in snake_factory.create_snakes({'auto': config.snake_count - 1}, start_length=config.start_length):
                self._snake_handler.add_snake(auto_snake)

        for snake in self._snake_handler.get_snakes():
            start_pos = self._snake_enviroment.add_snake(snake.get_id(), start_length=snake.get_length())
            snake.set_init_data(self._snake_enviroment.get_init_data().__dict__)
            snake.set_start_position(start_pos)

        self._loop.set_snake_handler(self._snake_handler)
        self._loop.set_environment(self._snake_enviroment)
        if default_config.max_steps is not None:
            self._loop.set_max_steps(default_config.max_steps)
        if default_config.max_no_food_steps is not None:
            self._loop.set_max_no_food_steps(default_config.max_no_food_steps)
        else:
            self._loop.set_max_no_food_steps((self._snake_enviroment.get_init_data().height * self._snake_enviroment.get_init_data().width) // 2)

    @init_check
    def add_observer(self, observer: ILoopObserver):
        """Adds an observer to the loop"""
        if not isinstance(observer, ILoopObserver):
            raise ValueError('Observer must be an instance of ILoopObserver')
        self._loop.add_observer(observer)

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
            init_data = self._snake_enviroment.get_init_data()
            run_data_observer = RunDataLoopObserver()
            run_data_observer.set_adapter(
                RunDataAdapter(
                    init_data,
                    create_color_map(init_data.snake_values)
                )
            )
            self._loop.add_observer(run_data_observer)
        run_data_observer.add_observer(observer)

    @init_check
    def get_init_data(self) -> EnvInitData:
        """Returns the initial data of the environment"""
        return self._snake_enviroment.get_init_data()

    def _initialize_controllers(self):
        """Initializes the controllers"""
        ctl_collection = ControllerCollection()
        for snake in self._snake_handler.get_snakes():
            if isinstance(snake, ManualSnake):
                ctl_collection.bind_controller(snake)
        ctl_collection.handle_controllers()

    def run(self, stop_event,config, *observers):
        """Starts the loop
        Args:
            stop_event: Event object to stop the loop
        """
        sim_config = None
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
        self.init(sim_config)
        if not config.no_record:
            recording_file = config.record_file if config.record_file else None
            self.add_run_data_observer(
            RecorderRunDataObserver(
                recording_dir=config.record_dir,
                recording_file=recording_file,
                as_proto=True
            )
            )
        self.add_run_data_observer(*observers)
        self._initialize_controllers()
        try:
            self._loop.start(stop_event)
        except Exception as e:
            self._loop.stop()
            log.exception(e)

    @init_check
    def stop(self):
        """Stops the loop"""
        self._loop.stop()

loop_control = SnakeLoopControl()
 
def setup_loop():
    return loop_control