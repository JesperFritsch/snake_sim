

from snake_sim.loop_observables.main_loop import SimLoop
from snake_sim.loop_observers.state_builder_observer import StateBuilderObserver
from snake_sim.loop_observers.map_builder_observer import MapBuilderObserver
from snake_sim.rl.environment.rl_snake_env import RLSnakeEnv
from snake_sim.rl.rl_data_queue import RLPendingTransitCache, RLMetaDataQueue
from snake_sim.rl.types import RLTransitionData, PendingTransition
from snake_sim.rl.reward import compute_rewards


class RLTrainingLoop(SimLoop):
    """Reinforcement learning training loop.

    Extends the main simulation loop with RL-specific functionality.
    """

    def __init__(self, current_episode: int = 0):
        super().__init__()
        self._current_episode = current_episode
        self._pending_transition_cache = RLPendingTransitCache()
        self._transition_queue = RLMetaDataQueue()
        self._previous_pending_transitions: dict[int, PendingTransition] = {}
        self._state_builder: StateBuilderObserver = StateBuilderObserver()
        self._map_builder: MapBuilderObserver = MapBuilderObserver()
        self.add_observer(self._state_builder)
        self.add_observer(self._map_builder)

    def _post_update(self):
        super()._post_update()
        snake_ids = set(self._previous_pending_transitions.keys())
        current_sim_state = self._state_builder.get_state(self._steps)
        current_sim_map = self._map_builder.get_map(self._steps)
        if self._steps > 0:
            prev_sim_state = self._state_builder.get_state(self._steps - 1)
            prev_sim_map = self._map_builder.get_map(self._steps - 1)
        
        rewards = compute_rewards(
            (prev_sim_state, prev_sim_map),
            (current_sim_state, current_sim_map),
            snake_ids,
        )   
        current_pending_transitions = self._pending_transition_cache.get_transitions()
        self._finalize_pending_transitions(current_pending_transitions, rewards)
        self._previous_pending_transitions = current_pending_transitions
        self._pending_transition_cache.clear()

    def _finalize_pending_transitions(self, current_pending_transitions: dict[int, PendingTransition], rewards: dict[int, float]) -> dict[int, PendingTransition]:
        """Gathers pending transitions from all snakes in the environment.

        Returns:
            A dictionary mapping snake IDs to their corresponding PendingTransition objects.
        """

        if self._previous_pending_transitions:
            for snake_id, pending_transition in self._previous_pending_transitions.items():
                current_pending = current_pending_transitions.get(snake_id, None)
                if current_pending is not None:
                    next_state = pending_transition.state
                    done=False
                else: # snake died
                    next_state = None
                    done=True
                transition = RLTransitionData(
                    transition_nr=self._steps,
                    state=pending_transition.state,
                    action_index=pending_transition.action_index,
                    reward=rewards[snake_id],
                    next_state=next_state,
                    snake_id=snake_id,
                    meta=pending_transition.meta,
                    done=done,
                    episode_id=self._current_episode,
                )
                self._transition_queue.add_transition(transition)
    