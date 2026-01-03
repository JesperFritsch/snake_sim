"""
RL data Queue / Bus for sharing transition data between agents and the trainer.
"""

import time

from abc import ABC, abstractmethod

from queue import Queue, Empty
from multiprocessing import Queue as MPQueue
from multiprocessing import queues as mp_queues
from snake_sim.utils import SingletonMeta
from snake_sim.rl.types import RLTransitionData, PendingTransition
from typing import Protocol, TypeVar, runtime_checkable



T = TypeVar("T")

@runtime_checkable
class QueueLike(Protocol[T]):
    def put(self, item: T) -> None: ...
    def get(self) -> T: ...
    def qsize(self) -> int: ...
    def empty(self) -> bool: ...


class RLMetaDataQueue:
    """Transition queue wrapper.

    Note: This is intentionally *not* a singleton.

    In single-process training you can still create exactly one instance and pass
    it around. For multi-process training (actors + learner) we *must* be able to
    create multiple independent wrappers, each bound to its own IPC queue.
    A singleton here can silently bind the wrong underlying queue and cause
    confusing drain behaviour.
    """

    def __init__(self, queue: QueueLike[RLTransitionData] | None = None):
        self.t_queue: QueueLike = queue if queue is not None else Queue()

    def add_transition(self, transition: RLTransitionData):
        """Add a transition to the queue.

        This is a non-blocking put; producer threads/processes enqueue and trainer drains.
        """
        self.t_queue.put(transition)

    def size(self) -> int:
        """Get the current size of the transition queue."""
        return self.t_queue.qsize()

    def get_transitions(self) -> list[RLTransitionData]:
        """Drain all currently queued transitions (FIFO order)."""
        transitions: list[RLTransitionData] = []

        # IMPORTANT: For `multiprocessing.Queue`, `.empty()` and `.qsize()` are not
        # reliable and may return stale values. The only correct way to drain is to
        # repeatedly attempt a non-blocking get until an Empty exception.
        # Drain until we observe emptiness for a short grace period.
        # This is robust to multiprocessing.Queue's feeder thread delay.
        while self.size() > 0:
            try:
                transitions.append(self.t_queue.get())
            except (Empty, mp_queues.Empty):
                break

        return transitions


class RLPendingTransitCache(metaclass=SingletonMeta):
    def __init__(self):
        self._pending_transitions: dict[int, PendingTransition] = {}
    
    def add_transition(self, pending_transition: PendingTransition):
        self._pending_transitions[pending_transition.snake_id] = pending_transition

    def clear(self):
        self._pending_transitions.clear()
    
    def get_transitions(self) -> dict[int, PendingTransition]:
        return self._pending_transitions