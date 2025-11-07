
from typing import Any, Optional
import numpy as np

from dataclasses import dataclass, field
import uuid


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
    step_nr: int
    state_buffer: np.ndarray
    action_index: int
    next_state_buffer: np.ndarray
    reward: float
    done: bool = False
    snake_id: Optional[str] = None
    episode_id: Optional[str] = None
    meta: Optional[RLMetaData] = None

