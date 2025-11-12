
import logging
import json
import time
from pathlib import Path

from importlib import resources as pkg_resources

from snake_sim.environment.food_handlers import FoodHandler
from snake_sim.environment.snake_env import SnakeEnv
from snake_sim.environment.types import DotDict, SnakeConfig, StrategyConfig
from snake_sim.environment.snake_handlers import SnakeHandler
from snake_sim.rl.loop_observables.rl_training_loop import RLTrainingLoop
from snake_sim.environment.snake_factory import SnakeFactory
from snake_sim.rl.types import TrainingConfig
from snake_sim.rl.ppo_trainer import PPOTrainer
from snake_sim.logging_setup import setup_logging


with pkg_resources.open_text('snake_sim.config', 'default_config.json') as config_file:
    default_config = DotDict(json.load(config_file))

setup_logging(log_level=logging.INFO)

log = logging.getLogger(Path(__file__).stem)


def setup_training_loop(config: TrainingConfig, episode_id: int = 0, snapshot_dir: str = None) -> RLTrainingLoop:
    snake_env = SnakeEnv(
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
    training_loop = RLTrainingLoop(current_episode=0)
    training_loop.set_environment(snake_env)
    training_loop.set_snake_handler(snake_handler)
    if config.max_steps_per_epoch is not None:
        training_loop.set_max_steps(config.max_steps_per_epoch)

    return training_loop


def add_snakes(snake_env: SnakeEnv, snake_handler: SnakeHandler, snapshot_dir: str = None):
    from snake_sim.rl.snakes.ppo_snake import PPOSnake
    from snake_sim.snakes.survivor_snake import SurvivorSnake
    from snake_sim.snakes.strategies.utils import apply_strategies
    
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
        init_pos = snake_env.add_snake(snake_id)
        snake.set_id(snake_id)
        snake.set_start_length(2)
        snake.set_start_position(init_pos)

    init_data = snake_env.get_init_data()
    for snake in snake_dict.values():
        snake.set_init_data(init_data)
    snake_handler.finalize(init_data)


def train(config: TrainingConfig):
    # Set up snapshot directory for model sharing
    snapshot_dir = "models/ppo_training"
    Path(snapshot_dir).mkdir(parents=True, exist_ok=True)
    
    trainer = PPOTrainer(snapshot_dir=snapshot_dir)
    trainer.start_background()
    
    log.info(f"Starting training for {config.epochs} episodes")
    log.info(f"Model snapshots will be saved to: {snapshot_dir}")
    
    try:
        for episode in range(config.epochs):
            episode_start_time = time.time()
            try:
                training_loop = setup_training_loop(config, episode_id=episode, snapshot_dir=snapshot_dir)
                
                # Track episode stats
                initial_queue_size = trainer.get_queue_size()
                initial_update_count = trainer.get_update_count()
                
                training_loop.start()
                
                episode_duration = time.time() - episode_start_time
                final_queue_size = trainer.get_queue_size()
                final_update_count = trainer.get_update_count()
                
                # Log episode completion stats
                transitions_generated = final_queue_size - initial_queue_size + (final_update_count - initial_update_count) * 256  # Estimate consumed
                if episode % 10 == 0 or episode < 10:  # Log more frequently at start
                    log.info(f"Episode {episode}/{config.epochs} completed in {episode_duration:.2f}s, "
                           f"generated ~{transitions_generated} transitions, "
                           f"training updates: {final_update_count}")
                
                # Log occasional training progress
                if episode % 50 == 0 and episode > 0:
                    log.info(f"Training Progress: {episode}/{config.epochs} episodes ({100*episode/config.epochs:.1f}%) completed, "
                           f"Total training updates: {final_update_count}")
                           
            except Exception as e:
                log.error(f"Episode {episode} failed: {e}", exc_info=True)
                # Continue with next episode instead of crashing
    
    except KeyboardInterrupt:
        log.info("Training interrupted by user")
    except Exception as e:
        log.error(f"Training failed: {e}", exc_info=True)
    finally:
        log.info("Training completed, stopping background trainer")
        trainer.stop_background()


if __name__ == "__main__":
    rl_config = TrainingConfig(
        epochs=10000,
        max_steps_per_epoch=5000
    )
    train(rl_config)