"""
RL data Queue / Bus for sharing transition data between agents and the trainer.
"""

from queue import Queue
from snake_sim.utils import SingletonMeta
from snake_sim.rl.types import RLTransitionData

class RLMetaDataQueue(metaclass=SingletonMeta):
    def __init__(self):
        self.transitions = Queue()

    def add_transition(self, transition: RLTransitionData):
        """Add a transition to the queue.

        This is a non-blocking put; producer threads/processes enqueue and trainer drains.
        """
        self.transitions.put(transition)

    def size(self) -> int:
        """Get the current size of the transition queue."""
        return self.transitions.qsize()

    def get_transitions(self) -> list[RLTransitionData]:
        """Drain all currently queued transitions (FIFO order)."""
        transitions = []
        while not self.transitions.empty():
            transitions.append(self.transitions.get())
        return transitions
