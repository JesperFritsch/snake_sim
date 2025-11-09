

from snake_sim.loop_observers.state_builder_observer import StateBuilderObserver
from snake_sim.environment.types import LoopStepData
from snake_sim.rl.types import State, PendingTransition, RLTransitionData
from snake_sim.rl.rl_data_queue import RLPendingTransitCache, RLMetaDataQueue


class TransitionCreatorObserver(StateBuilderObserver):
    """ Observer that creates transitions from step data and the states constructed by the StateBuilderObserver. 
        This observer is used instead of a subclass of the main loop because it has easy access to previous env states.
    """

    def __init__(self):
        super().__init__()
        self.pending_transition_cache = RLPendingTransitCache()
        self._previous_pending_transitions: dict[int, PendingTransition] = {}
        self._transition_queue = RLMetaDataQueue()

    def notify_step(self, step_data: LoopStepData):
        super().notify_step(step_data)
        current_pending_transitions = self.pending_transition_cache.get_transitions()
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
                    state=pending_transition.state,
                    action_index=pending_transition.action_index,
                    next_state=next_state,
                    reward=step_data.rewards.get(snake_id, 0.0),
                    done=done,
                    snake_id=snake_id,
                    meta=pending_transition.meta,
                    episode_id=None
                )
                self._transition_queue.add_transition(transition)
        self._previous_pending_transitions = current_pending_transitions