from typing import Union
import json
from importlib import resources as pkg_resources

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
    verbose: int


@dataclass
class GameConfig(SimConfig):
    player_count: int
    spm: int


class SnakeLoopControl:
    def __init__(self):
        self.loop = None

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
            self.loop = GameLoop()
            self.loop.set_steps_per_min(config.spm)
            for snake_config in default_config.snake_configs[:config.player_count]:
                man_snake = snake_factory.create_snake(**snake_config['snake'], help=1)
                ctl_collection.bind_controller(man_snake)
                snake_handler.add_snake(man_snake, **snake_config['env'])
            for snake_config in default_config.snake_configs[config.player_count:config.snake_count]:
                snake_handler.add_snake(
                    snake_factory.create_snake('manual', **snake_config['snake'], calc_timeout=config.calc_timeout),
                    **snake_config['env']
                )
            ctl_collection.handle_controllers()
        else:
            # Initialize simulation loop
            self.loop = SimLoop()
            for snake_config in default_config.snake_configs[:config.snake_count]:
                snake_handler.add_snake(
                    snake_factory.create_snake('auto', **snake_config['snake'], calc_timeout=config.calc_timeout),
                    **snake_config['env']
                )

        for snake in snake_handler.snakes.values():
            snake_enviroment.add_snake(snake.get_id(), start_position=None, start_length=snake.get_length())
        self.loop.set_snake_handler(snake_handler)
        self.loop.set_environment(snake_enviroment)
        # self.loop.set_max_steps(default_config.MAX_STEPS)
        self.loop.set_max_no_food_steps((snake_enviroment.get_init_data().height * snake_enviroment.get_init_data().width) // 2)
        # self.loop.add_observer()


    def run(self):
        self.loop.start()