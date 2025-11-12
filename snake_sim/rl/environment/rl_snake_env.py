import numpy as np

from snake_sim.environment.snake_env import SnakeEnv
from snake_sim.environment.types import Coord


class RLSnakeEnv(SnakeEnv):
    """Reinforcement Learning Snake Environment.

    This environment is designed for training RL agents to play the Snake game.
    It extends the base SnakeEnv with RL-specific functionalities.
    """

    def __init__(self, width, height, free_value, blocked_value, food_value):
        super().__init__(width, height, free_value, blocked_value, food_value)
        # Additional RL-specific initialization can be added here

    def reset(self) -> dict[int, Coord]:
        """Reset the environment to the initial state for a new episode."""
        self._map = np.copy(self._base_map)
        for snake_rep in self._snake_reps.values():
            new_position = self._random_free_tile()
            snake_rep.reset(start_position=new_position)
            self._place_snake_on_map(snake_rep)
        return {id: snake_rep.get_head() for id, snake_rep in self._snake_reps.items()}
