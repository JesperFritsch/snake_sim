
from typing import Any, Optional
import numpy as np

from dataclasses import dataclass, field
import uuid


class State:
    """Container for snake state representation used in RL.

    Fields:
        map: np.ndarray representing the environment state (e.g., channels x height x width).
        ctx: Optional[np.ndarray] representing additional context (e.g., snake length ratio).
    """
    def __init__(self, map: np.ndarray, ctx: Optional[np.ndarray] = None):
        self.map = map
        self.ctx = ctx
        self.meta: dict[str, Any] = {}  # Additional metadata if needed


class RLMetaData:
    """Base class for RL meta data associated with transitions."""
    pass


@dataclass
class PPOMetaData(RLMetaData):
        """Meta data specific to PPO (only immutable per-step inference outputs).

        Fields:
            log_prob: log probability of the taken action under the OLD policy.
            value_estimate: critic V(s) under the OLD parameters.

        Advantage and return targets are stored externally mapped via transition_id to
        keep this object immutable regarding trainer-derived quantities.
        """
        log_prob: float
        value_estimate: float


@dataclass
class RLTransitionData:
    """Generic transition data used by trainers.

    Added snake_id and episode_id for multi-agent grouping and trajectory segmentation.
    transition_id provides a stable key for externally stored trainer annotations
    (e.g., advantage, return) without mutating the meta object.
    """
    transition_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    transition_nr: int
    state: State
    action_index: int
    reward: float
    snake_id: int
    meta: RLMetaData
    done: bool = False
    next_state: Optional[State] = None
    episode_id: Optional[str] = None

@dataclass
class PendingTransition:
    state: State
    action_index: int
    meta: PPOMetaData
    transition_nr: int
    snake_id: int