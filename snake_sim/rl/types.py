
import numpy as np
import uuid
from typing import Any, Optional
from dataclasses import dataclass, field

from snake_sim.environment.types import SimConfig


class State:
    """Container for snake state representation used in RL.

    Fields:
        map: np.ndarray representing the environment state (e.g., channels x height x width).
        ctx: Optional[np.ndarray] representing additional context (e.g., snake length ratio).
    """
    def __init__(self, map: np.ndarray, ctx: Optional[np.ndarray] = None, meta: Optional[dict[str, Any]] = None):
        self.map = map
        self.ctx = ctx
        self.meta = meta if meta is not None else {}


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
    transition_nr: int
    process_id: int
    state: State
    action_index: int
    reward: float
    snake_id: int
    next_state: State
    meta: RLMetaData
    episode_id: int
    done: bool = False
    # Explicit behavior event: whether the snake ate food on this transition.
    # This should be set by the environment/loop (source of truth), not inferred
    # from total reward magnitude (since reward shaping can change).
    ate_food: bool = False
    transition_id: str = field(default_factory=lambda: uuid.uuid4().hex)


@dataclass
class PendingTransition:
    state: State
    action_index: int
    meta: RLMetaData
    transition_nr: int
    snake_id: int


@dataclass
class RLTrainingConfig:
    episodes: int
    food_tiles: int
    nr_snakes: int
    max_steps_per_episode: Optional[int] = None
    training_maps: list[str] = field(default_factory=list)
    max_no_food_steps: Optional[int] = None