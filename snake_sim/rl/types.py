
from typing import Any, Optional

from dataclasses import dataclass, field


class RLMetaData:
    """Base class for RL meta data associated with transitions."""
    pass


@dataclass
class PPOMetaData(RLMetaData):
    log_prob: float
    value_estimate: float
    advantage: float
    return_estimate: float


@dataclass
class RLTransitionData:
    step_nr: int
    state_buffer: Any
    action_index: int
    next_state_buffer: Any
    reward: float
    done: bool
    meta: Optional[RLMetaData] = field(default_factory=dict)

