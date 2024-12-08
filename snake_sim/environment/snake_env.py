from typing import Union
import json
from importlib import resources as pkg_resources

from snake_sim.environment.main_loop import SimLoop, GameLoop, FoodHandler
from snake_sim.environment.snake_handlers import SnakeHandler
from snake_sim.controllers.keyboard_controller import ControllerCollection
from snake_sim.environment.snake_factory import SnakeFactory
from snake_sim.utils import get_map_files_mapping, DotDict
from dataclasses import dataclass

with pkg_resources.open_text('snake_sim.config', 'default_config.json') as config_file:
    default_config = DotDict(json.load(config_file))

@dataclass
class SimConfig:
    map: str
    food: int
    food_decay: int
    snake_count: int
    calc_timeout: int
    verbose: int


@dataclass
class GameConfig(SimConfig):
    player_count: int
    spm: int


class SnakeEnv:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.loop = None

    def init(self, config: Union[SimConfig, GameConfig]):
        """Initializes the environment with the given configuration"""
        snake_handler = SnakeHandler()
        snake_factory = SnakeFactory()
        if isinstance(config, GameConfig):
            ctl_collection = ControllerCollection()
            self.loop = GameLoop()
            self.loop.set_steps_per_min(config.spm)
            for snake_config in default_config.snake_configs[:config.player_count]:
                man_snake = snake_factory.create_snake(**snake_config['snake'], help=1)
                ctl_collection.bind_controller(man_snake)
                snake_handler.add_snake(man_snake, **snake_config['env'])
            for snake_config in default_config.snake_configs[config.player_count:config.snake_count]:
                snake_handler.add_snake(snake_factory.create_snake(**snake_config['snake'], calc_timeout=config.calc_timeout), **snake_config['env'])
            ctl_collection.handle_controllers()
        else:
            self.loop = SimLoop()
            for snake_config in default_config.snake_configs[:config.snake_count]:
                snake_handler.add_snake(snake_factory.create_snake(**snake_config['snake'], calc_timeout=config.calc_timeout), **snake_config['env'])
        self.loop.init(self.width, self.height)
        if config.map:
            self.load_map(config.map)
        self.loop.set_food_handler(FoodHandler(self.width, self.height, config.food, config.food_decay))

    def run(self):
        self.loop.start()

    def load_map(self, map_name):
        files_mapping = get_map_files_mapping()
        if file_path := files_mapping.get(map_name):
            self.loop.load_map(file_path)
        else:
            raise FileNotFoundError(f"Map {map_name} not found")