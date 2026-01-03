
from abc import abstractmethod

import snake_sim.debugging as debug

from snake_sim.environment.types import Coord, EnvStepData
from snake_sim.snakes.snake_base import SnakeBase
from snake_sim.map_utils.general import print_map
from snake_sim.rl.types import State
from snake_sim.rl.state_builder import (
    BaseStateBuilder, 
    CompleteStateBuilder, 
    DirectionHintsAdapter,
    ActionMaskAdapter,
    SnakeContext
)


# debug.activate_debug()
# debug.enable_debug_for("RLSnakeBase")


class RLSnakeBase(SnakeBase):
    """ Base class for RL controlled snakes. """

    def __init__(self):
        super().__init__()
        # Build state with direction hints + hard action masking.
        self._state_builder = CompleteStateBuilder(
            BaseStateBuilder(),
            adapters=[DirectionHintsAdapter(), ActionMaskAdapter()],
        )

    def _get_state(self, step_data: EnvStepData) -> State:
        snake_ctx = SnakeContext(
            snake_id=self._id,
            head=self._head_coord,
            body_coords=list(self._body_coords),
            length=self._length
        ) 
        return self._state_builder.build(
            self._env_meta_data,
            step_data,
            snake_ctx
        )

    def _next_step(self) -> Coord:
        state = self._get_state(self._env_step_data)
        debug.debug_print(f"RL Snake: {self._id} state ctx: {state.ctx}, meta: {state.meta}")
        return self._next_step_for_state(state)

    @abstractmethod
    def _next_step_for_state(self, state: State) -> Coord:
        """ To be implemented by subclasses. Given a state, return the next step coordinate. """
        pass