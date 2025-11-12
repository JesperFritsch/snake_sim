
from typing import Dict, List, Set, Deque
from collections import deque
from snake_sim.environment.types import (
    LoopStartData,
    Coord,
    CompleteStepState
)

from snake_sim.loop_observers.consumer_observer import ConsumerObserver


class NoMoreSteps(Exception):
    pass


class CurrentIsFirst(Exception):
    pass


class StateBuilderObserver(ConsumerObserver):
    """ Receives loop data and can construct framebuffers of the simulation. does not keep all frames in memory, only the current one.
    Creates new frames either next or previous from the current one. """
    def __init__(self):
        super().__init__()
        self._snake_bodies: Dict[int, Deque] = {}
        self._current_step_idx = 0
        self._current_state: CompleteStepState = None

    def notify_start(self, start_data: LoopStartData):
        super().notify_start(start_data)
        init_data = self._start_data.env_meta_data
        self._current_state: CompleteStepState = CompleteStepState(
            env_meta_data=init_data,
            food=set(),
            snake_bodies={}
        )
        for s_id, pos in init_data.start_positions.items():
            self._current_state.snake_bodies[s_id] = deque([pos])

    def get_state(self, state_idx: int):
        self._goto_state(state_idx)
        return self.get_current_state()

    def get_current_state(self) -> CompleteStepState:
        self._current_state.food = set(map(lambda f: Coord(*f), self._current_state.food))
        return self._current_state

    def get_next_state(self):
        self._goto_next_state()
        return self.get_current_state()

    def get_prev_state(self):
        self._goto_prev_state()
        return self.get_current_state()

    def _goto_state(self, state_idx: int):
        idx_delta = state_idx - self._current_step_idx
        while idx_delta != 0:
            if idx_delta > 0:
                self._goto_next_state()
                idx_delta -= 1
            else:
                self._goto_prev_state()
                idx_delta += 1

    def _goto_next_state(self):
        if self._current_step_idx >= len(self._steps):
            if self._stop_data is not None:
                raise StopIteration("No more states available")
            raise NoMoreSteps("Need to receive more steps to generate states")
        self._current_state.state_idx += 1
        step_data = self._steps[self._current_step_idx]
        self._current_step_idx += 1
        self._current_state.food.update(step_data.new_food)
        self._current_state.food.difference_update(step_data.removed_food)
        for s_id, dir in step_data.decisions.items():
            body = self._current_state.snake_bodies[s_id]
            new_head = body[0] + dir
            body.appendleft(new_head)
            tail_dir = step_data.tail_directions[s_id]
            if tail_dir != (0, 0):
                body.pop()

    def _goto_prev_state(self):
        self._current_step_idx -= 1
        if self._current_step_idx < 0:
            raise CurrentIsFirst()
        self._current_state.state_idx -= 1
        curr_step_data = self._steps[self._current_step_idx]
        self._current_state.food.difference_update(curr_step_data.new_food)
        self._current_state.food.update(curr_step_data.removed_food)
        for s_id, tail_dir in curr_step_data.tail_directions.items():
            body = self._current_state.snake_bodies[s_id]
            popped_tile = body.popleft()
            if tail_dir != (0, 0):
                old_tail = body[-1] - tail_dir if len(body) > 1 else popped_tile - tail_dir
                body.append(old_tail)

