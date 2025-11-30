
import logging
import time
import random
from pathlib import Path
from dataclasses import dataclass

import snake_sim.debugging as debug
from snake_sim.loop_observables.main_loop import SimLoop
from snake_sim.environment.types import LoopStepData
from snake_sim.rl.environment.rl_snake_env import RLSnakeEnv
from snake_sim.rl.training.rl_data_queue import RLPendingTransitCache, RLMetaDataQueue
from snake_sim.rl.types import RLTransitionData, PendingTransition, RLTrainingConfig
from snake_sim.rl.state_builder import print_state 
from snake_sim.rl.reward import compute_rewards
from snake_sim.map_utils.general import print_map

log = logging.getLogger(Path(__file__).stem)


class RLTrainingLoop(SimLoop):
    """Reinforcement learning training loop.

    Extends the main simulation loop with RL-specific functionality.
    """

    def __init__(self, config: RLTrainingConfig):
        super().__init__()
        self._config = config
        self._env: RLSnakeEnv = None
        self._current_episode = 0
        self._pending_transition_cache = RLPendingTransitCache()
        self._transition_queue = RLMetaDataQueue()
        self._previous_pending_transitions: dict[int, PendingTransition] = {}
        self._prev_sim_state = None
        self._prev_sim_map = None
        
        # Training progress tracking
        self._episode_start_time = None
        self._initial_transitions_count = 0

    def set_environment(self, env: RLSnakeEnv):
        if not isinstance(env, RLSnakeEnv):
            raise ValueError('Environment must be an instance of RLSnakeEnv')
        return super().set_environment(env)

    def _pre_update(self):
        super()._pre_update()
        if debug.is_debug_active():
            print(f"Step {self._steps} starting: ==============================")

    def _post_update(self):
        super()._post_update()
        current_pending_transitions = self._pending_transition_cache.get_transitions()
        snake_ids = set(current_pending_transitions.keys())
        if not snake_ids:
            return
        current_sim_state = self._env.get_state()
        current_sim_map = self._env.get_map()
        current_sim_state.state_idx = self._steps
        
        if not self._prev_sim_state is None and not self._prev_sim_map is None:
            rewards = compute_rewards(
                (self._prev_sim_state, self._prev_sim_map),
                (current_sim_state, current_sim_map),
                snake_ids,
            )   
            self._finalize_pending_transitions(current_pending_transitions, rewards)
        self._prev_sim_state = current_sim_state
        self._prev_sim_map = current_sim_map
        self._previous_pending_transitions = current_pending_transitions
        self._pending_transition_cache.clear()

    def _finalize_pending_transitions(self, current_pending_transitions: dict[int, PendingTransition], rewards: dict[int, float]) -> dict[int, PendingTransition]:
        """Gathers pending transitions from all snakes in the environment.

        Returns:
            A dictionary mapping snake IDs to their corresponding PendingTransition objects.
        """

        if self._previous_pending_transitions:
            for snake_id, pre_pending_transition in self._previous_pending_transitions.items():
                current_pending = current_pending_transitions.get(snake_id, None)
                if current_pending is not None and self._env.snake_is_alive(snake_id):
                    # Use the current environment state, not the pending transition's state
                    # The pending transition's state is what the snake will use for the NEXT decision
                    next_state = current_pending.state
                    done=False
                else: # snake died
                    next_state = None
                    done=True
                transition = RLTransitionData(
                    transition_nr=self._steps,
                    state=pre_pending_transition.state,
                    action_index=pre_pending_transition.action_index,
                    reward=rewards[snake_id],
                    next_state=next_state,
                    snake_id=snake_id,
                    meta=pre_pending_transition.meta,
                    done=done,
                    episode_id=self._current_episode,
                )
                self._transition_queue.add_transition(transition)
                if debug.is_debug_active():
                    print(f"RL Training Loop: Step {self._steps} completed. Rewards: {rewards}")
                    self._print_env_map()
                    print("From state:")
                    print_state(transition.state)
                    if transition.next_state is None:
                        print("To state: <DEAD>")
                    else:
                        print("To state:")
                        print_state(transition.next_state)
                    print(f"Action taken: {transition.action_index}, Reward: {transition.reward}, Done: {transition.done}")

    def _get_map_path_from_selection(self) -> str:
        """Selects a training map from the configured list."""
        if not self._config.training_map_paths:
            return None
        return random.choice(self._config.training_map_paths) or None

    def _reset(self):
        self._steps = 0
        self._current_step_data: LoopStepData = None
        self._step_start_time = None
        self._is_running = False
        self._did_notify_start = False
        self._did_notify_stop = False
        self._prev_sim_state = None
        self._prev_sim_map = None
        next_map_path = self._get_map_path_from_selection()
        if next_map_path:
            self._env.load_map(next_map_path)
        else:
            self._env.clear_map()
        new_positions = self._env.reset()
        self._snake_handler.reset(new_positions)
        for observer in self._observers:
            observer.reset()
        self._previous_pending_transitions.clear()
        self._pending_transition_cache.clear()

    def start(self):
        log.info(f"Starting RL training for {self._config.episodes} episodes")
        for episode in range(self._config.episodes):
            self._current_episode = episode
            self._episode_start_time = time.time()
            
            # Run the episode
            super().start()
            
            # Calculate episode metrics
            episode_duration = time.time() - self._episode_start_time
            
            # Log episode completion
            if episode % 10 == 0 or episode < 10:  # Log more frequently at start
                log.info(f"Episode {episode}/{self._config.episodes} completed in {episode_duration:.2f}s, "
                       f"steps: {self._steps}")
            
            # Log progress milestones
            if episode % 50 == 0 and episode > 0:
                progress_pct = 100 * episode / self._config.episodes
                log.info(f"🎯 Training Progress: {episode}/{self._config.episodes} episodes "
                       f"({progress_pct:.1f}%) completed")
            
            # Reset for next episode
            self._reset()
            
        log.info(f"✅ RL training completed! Finished {self._config.episodes} episodes")

    def _print_env_map(self):
        state = self._env.get_state()
        self._env.print_map()
        for snake_id, snake in state.snake_bodies.items():
            head = snake[0]
            print(f"Snake {snake_id} head at: ({head.x}, {head.y})")