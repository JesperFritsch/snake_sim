import numpy as np

from snake_sim.environment.snake_env import SnakeEnv
from snake_sim.environment.types import Coord, CompleteStepState


class RLSnakeEnv(SnakeEnv):
    """Reinforcement Learning Snake Environment.

    This environment is designed for training RL agents to play the Snake game.
    It extends the base SnakeEnv with RL-specific functionalities.
    """

    def __init__(self, width, height, free_value, blocked_value, food_value):
        super().__init__(width, height, free_value, blocked_value, food_value)
        # Additional RL-specific initialization can be added here

    def snake_is_alive(self, snake_id: int) -> bool:
        """Check if the snake with the given ID is alive."""
        snake_rep = self._snake_reps.get(snake_id)
        return snake_rep.is_alive if snake_rep else False

    def clear_map(self):
        """Clear the environment map."""
        self._map.fill(self._free_value)

    def get_state(self, for_id: int = None) -> CompleteStepState:
        """Get the complete step state of the environment."""
        return CompleteStepState(
            env_meta_data=self.get_init_data(),
            food=set(self.get_food()),
            snake_bodies={id: snake_rep.body.copy() for id, snake_rep in self._snake_reps.items()},
            snake_alive={id: snake_rep.is_alive for id, snake_rep in self._snake_reps.items()},
            snake_ate={id: snake_rep.get_head() in set(self.get_food()) for id, snake_rep in self._snake_reps.items()},
            state_idx=0
        )

    def reset(self) -> dict[int, Coord]:
        """Reset the environment to the initial state for a new episode."""
        self._food_handler.clear()
        self._map = self.get_base_map()
        for snake_rep in self._snake_reps.values():
            new_position = self._random_free_tile()
            snake_rep.reset(start_position=new_position)
            self._place_snake_on_map(snake_rep)
        return {id: snake_rep.get_head() for id, snake_rep in self._snake_reps.items()}
