
import logging
import json
import time

from pathlib import Path

import importlib.util as imp_util
from importlib import resources as pkg_resources

from snake_sim.environment.food_handlers import FoodHandler
from snake_sim.environment.types import DotDict, SnakeConfig, StrategyConfig
from snake_sim.environment.snake_handlers import SnakeHandler
from snake_sim.environment.snake_factory import SnakeFactory

from snake_sim.rl.snakes.ppo_snake import PPOSnake
from snake_sim.snakes.survivor_snake import SurvivorSnake
from snake_sim.snakes.strategies.utils import apply_strategies

from snake_sim.map_utils.general import get_map_files_mapping
from snake_sim.loop_observers.file_persist_observer import FilePersistObserver
from snake_sim.rl.loop_observables.rl_training_loop import RLTrainingLoop
from snake_sim.rl.environment.rl_snake_env import RLSnakeEnv
from snake_sim.rl.types import RLTrainingConfig
from snake_sim.rl.ppo_trainer import PPOTrainer
from snake_sim.logging_setup import setup_logging


with pkg_resources.open_text('snake_sim.config', 'default_config.json') as config_file:
    default_config = DotDict(json.load(config_file))

PACKAGE_ROOT = Path(imp_util.find_spec('snake_sim').origin).parent

setup_logging(log_level=logging.INFO)

log = logging.getLogger(Path(__file__).stem)


def setup_training_loop(config: RLTrainingConfig, snapshot_dir: str = None) -> RLTrainingLoop:
    snake_env = RLSnakeEnv(
        width=32,
        height=32,
        free_value=default_config.free_value,
        blocked_value=default_config.blocked_value,
        food_value=default_config.food_value
    )
    food_handler = FoodHandler(
            32,
            32,
            25,
            0)
    snake_env.set_food_handler(food_handler)
    snake_handler = SnakeHandler()
    add_snakes(snake_env, snake_handler, snapshot_dir)
    training_loop = RLTrainingLoop(config)
    training_loop.set_environment(snake_env)
    training_loop.set_snake_handler(snake_handler)
    if config.max_steps_per_episode is not None:
        training_loop.set_max_steps(config.max_steps_per_episode)

    return training_loop


def add_snakes(snake_env: RLSnakeEnv, snake_handler: SnakeHandler, snapshot_dir: str = None):
    
    # Create PPO snakes with snapshot directory
    ppo_snakes = []
    for _ in range(1):
        snake = PPOSnake(
            snapshot_dir=snapshot_dir,
            poll_interval=2.0,  # Check for updates every 2 seconds
            auto_reload=True,
            eager_first_load=True,  # Load initial weights immediately
        )
        ppo_snakes.append(snake)
    
    # Create regular survivor snakes
    regular_snakes = []
    for _ in range(0):
        snake = SurvivorSnake()
        # Apply food seeker strategy
        snake_config = SnakeConfig(
            type='survivor',
            strategies={1: StrategyConfig(type='food_seeker')},
        )
        apply_strategies(snake, snake_config)
        regular_snakes.append(snake)

    for snake in ppo_snakes + regular_snakes:
        snake_handler.add_snake(snake)
    snake_dict = snake_handler.get_snakes()

    for snake_id, snake in snake_dict.items():
        init_pos = snake_env.add_snake(snake_id, start_length=2)
        snake.set_id(snake_id)
        snake.set_start_length(2)
        snake.set_start_position(init_pos)

    init_data = snake_env.get_init_data()
    for snake in snake_dict.values():
        snake.set_init_data(init_data)
    snake_handler.finalize(init_data)


def train(config: RLTrainingConfig):
    # Set up snapshot directory for model sharing
    snapshot_dir = "models/ppo_training_new"
    Path(snapshot_dir).mkdir(parents=True, exist_ok=True)
    
    trainer = PPOTrainer(snapshot_dir=snapshot_dir)
    trainer.start_background()
    
    log.info(f"Starting training for {config.episodes} episodes")
    log.info(f"Model snapshots will be saved to: {snapshot_dir}")
    training_loop = setup_training_loop(config, snapshot_dir=snapshot_dir)
    # file_persist_observer = FilePersistObserver(store_dir=Path(PACKAGE_ROOT) / "rl/trainings_runs")
    # training_loop.add_observer(file_persist_observer)
    try:
        training_loop.start()
    
    except KeyboardInterrupt:
        log.info("Training interrupted by user")
    except Exception as e:
        log.error(f"Training failed: {e}", exc_info=True)
    finally:
        log.info("Training completed, stopping background trainer")
        trainer.stop_background()
        training_loop.close()


if __name__ == "__main__":

    maps_mapping = get_map_files_mapping()

    training_maps = [
        "comps2",
        "comps",
        "lil_sign",
        "face",
        "patterns",
        "quarters3",
        "wavy",
        "tricky",
        "items",
        None
    ]

    map_paths = [maps_mapping.get(map_name) for map_name in training_maps]

    rl_config = RLTrainingConfig(
        episodes=500000,
        max_steps_per_episode=5000,
        training_map_paths=map_paths
    )
    train(rl_config)