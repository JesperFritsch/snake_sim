import json
import logging
from typing import Union, List
from importlib import resources as pkg_resources

from snake_sim.environment.interfaces.main_loop_interface import ILoopObserver

from snake_sim.environment.main_loop import SimLoop, GameLoop
from snake_sim.environment.snake_handlers import SnakeHandler
from snake_sim.controllers.keyboard_controller import ControllerCollection
from snake_sim.environment.snake_factory import SnakeFactory
from snake_sim.environment.snake_env import SnakeEnv, EnvInitData
from snake_sim.environment.food_handlers import FoodHandler
from snake_sim.utils import get_map_files_mapping, DotDict
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
            print(map_files_mapping)
            if config.map not in map_files_mapping:
                raise ValueError(f'Map {config.map} not found')
            self._snake_enviroment.load_map(map_files_mapping[config.map])

        if isinstance(config, GameConfig):
            # Initialize game loop add keyboard controllers
            ctl_collection = ControllerCollection()
            self._loop = GameLoop()
            self._loop.set_steps_per_min(config.spm)
            for man_snake in snake_factory.create_snakes({'manual': config.player_count}, start_length=config.start_length, help=1):
                ctl_collection.bind_controller(man_snake)
                self._snake_handler.add_snake(man_snake)

            for auto_snake in snake_factory.create_snakes({'auto': config.snake_count - config.player_count}, start_length=config.start_length):
                self._snake_handler.add_snake(auto_snake)
            ctl_collection.handle_controllers()

        else:
            # Initialize simulation loop
            self._loop = SimLoop()
            for auto_snake in snake_factory.create_snakes({'auto': config.snake_count}, start_length=config.start_length):
                self._snake_handler.add_snake(auto_snake)

        for snake in self._snake_handler.get_snakes():
            start_pos = self._snake_enviroment.add_snake(snake.get_id(), start_length=snake.get_length())
            snake.set_init_data(self._snake_enviroment.get_init_data().__dict__)
            snake.set_start_position(tuple(start_pos))

        self._loop.set_snake_handler(self._snake_handler)
        self._loop.set_environment(self._snake_enviroment)
        # self._loop.set_max_steps(default_config.MAX_STEPS)
        self._loop.set_max_no_food_steps((self._snake_enviroment.get_init_data().height * self._snake_enviroment.get_init_data().width) // 2)

    def get_snake_ids(self) -> List[int]:
        """Returns the ids of the snakes"""
        if not self._loop:
            raise ValueError('Loop not initialized')
        return [s.get_id() for s in self._snake_handler.get_snakes()]

    def add_observer(self, observer: ILoopObserver):
        """Adds an observer to the loop"""
        if not self._loop:
            raise ValueError('Loop not initialized')
        if not isinstance(observer, ILoopObserver):
            raise ValueError('Observer must be an instance of ILoopObserver')
        self._loop.add_observer(observer)

    def get_init_data(self) -> EnvInitData:
        """Returns the initial data of the environment"""
        if not self._loop:
            raise ValueError('Loop not initialized')
        return self._snake_enviroment.get_init_data()

    def run(self, stop_event):
        """Starts the loop
        Args:
            stop_event: Event object to stop the loop
        """
        try:
            if not self._loop:
                raise ValueError('Loop not initialized')
            self._loop.start(stop_event)
        except Exception as e:
            self._loop.stop()
            log.exception(e)

    def stop(self):
        """Stops the loop"""
        if not self._loop:
            raise ValueError('Loop not initialized')
        self._loop.stop()