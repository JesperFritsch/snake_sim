from typing import Union, Optional
import json
from importlib import resources as pkg_resources

from multiprocessing import Process, Pipe, Event

from snake_sim.environment.interfaces.main_loop_interface import ILoopObserver

from snake_sim.environment.main_loop import SimLoop, GameLoop
from snake_sim.environment.snake_handlers import SnakeHandler
from snake_sim.controllers.keyboard_controller import ControllerCollection
from snake_sim.environment.snake_factory import SnakeFactory
from snake_sim.environment.snake_env import SnakeEnv
from snake_sim.environment.food_handlers import FoodHandler
from snake_sim.utils import get_map_files_mapping, DotDict
from dataclasses import dataclass

with pkg_resources.open_text('snake_sim.config', 'default_config.json') as config_file:
    default_config = DotDict(json.load(config_file))


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
        snake_handler = SnakeHandler()
        snake_factory = SnakeFactory()
        snake_enviroment = SnakeEnv(config.width, config.height)
        food_handler = FoodHandler(config.width, config.height, config.food, config.food_decay)
        snake_enviroment.set_food_handler(food_handler)

        if config.map:
            map_files_mapping = get_map_files_mapping()
            if config.map not in map_files_mapping:
                raise ValueError(f'Map {config.map} not found')
            snake_enviroment.load_map(map_files_mapping[config.map])

        if isinstance(config, GameConfig):
            # Initialize game loop add keyboard controllers
            ctl_collection = ControllerCollection()
            self._loop = GameLoop()
            self._loop.set_steps_per_min(config.spm)
            for snake_config in default_config.snake_configs[:config.player_count]:
                man_snake = snake_factory.create_snake('manual', **snake_config['snake'], help=1)
                ctl_collection.bind_controller(man_snake)
                snake_handler.add_snake(man_snake)
            for snake_config in default_config.snake_configs[config.player_count:config.snake_count]:
                snake_handler.add_snake(
                    snake_factory.create_snake('auto', **snake_config['snake'], calc_timeout=config.calc_timeout)
                )
            ctl_collection.handle_controllers()
        else:
            # Initialize simulation loop
            self._loop = SimLoop()
            for snake_config in default_config.snake_configs[:config.snake_count]:
                snake_handler.add_snake(
                    snake_factory.create_snake('auto', **snake_config['snake'], calc_timeout=config.calc_timeout)
                )

        for snake in snake_handler.get_snakes():
            snake_enviroment.add_snake(snake.get_id(), start_length=snake.get_length())
            snake.set_init_data(dict(snake_enviroment.get_init_data()))

        self._loop.set_snake_handler(snake_handler)
        self._loop.set_environment(snake_enviroment)
        # self._loop.set_max_steps(default_config.MAX_STEPS)
        self._loop.set_max_no_food_steps((snake_enviroment.get_init_data().height * snake_enviroment.get_init_data().width) // 2)


    def add_observer(self, observer: ILoopObserver):
        """Adds an observer to the loop"""
        if not self._loop:
            raise ValueError('Loop not initialized')
        if not isinstance(observer, ILoopObserver):
            raise ValueError('Observer must be an instance of ILoopObserver')
        self._loop.add_observer(observer)

    def run(self, stop_event: Event):
        """Starts the loop"""
        if not self._loop:
            raise ValueError('Loop not initialized')
        self._loop.start(stop_event)