import time

from typing import Dict, List, Set, Deque
from collections import deque

from threading import Condition

from snake_sim.environment.types import (
    LoopStartData,
    Coord,
    CompleteStepState,
    LoopStepData,
    NoMoreSteps,
    CurrentIsFirst,
)

from snake_sim.loop_observers.consumer_observer import ConsumerObserver


class StateBuilderObserver(ConsumerObserver):
    """ Receives loop data and can construct framebuffers of the simulation. does not keep all frames in memory, only the current one.
    Creates new frames either next or previous from the current one. """
    def __init__(self):
        super().__init__()
        self._current_step_idx = 0
        self._current_state: CompleteStepState = None
        self._new_step_condition = Condition()

    def notify_start(self, start_data: LoopStartData):
        super().notify_start(start_data)
        init_data = self._start_data.env_meta_data
        self._current_state: CompleteStepState = CompleteStepState(
            env_meta_data=init_data,
            food=set(),
            snake_alive={},
            snake_bodies={},
            snake_ate={},
        )
        for s_id, pos in init_data.start_positions.items():
            self._current_state.snake_bodies[s_id] = deque([pos])

    def notify_step(self, step_data: LoopStepData):
        super().notify_step(step_data)
        with self._new_step_condition:
            self._new_step_condition.notify_all()

    def notify_stop(self, stop_data):
        super().notify_stop(stop_data)
        with self._new_step_condition:
            self._new_step_condition.notify_all()

    def _wait_for_state(self, idx: int):
        with self._new_step_condition:
            self._new_step_condition.wait_for(
                lambda: len(self._steps) > idx or self._stop_data is not None
            )

    def reset(self):
        self._current_step_idx = 0
        self._current_state = None
        return super().reset()

    def get_state(self, state_idx: int) -> CompleteStepState:
        self._goto_state(state_idx)
        return self.get_current_state()

    def peek_state(self, state_idx: int) -> CompleteStepState:
        """ Like get_state, but returns the live state instead of a copy.

        Cheaper than get_state (which rebuilds every body deque), so it suits
        consumers that read one state per rendered frame. The caller must treat
        the result as read-only and must not hold on to it across calls.
        """
        self._goto_state(state_idx)
        return self._current_state

    def get_max_state_idx(self) -> int:
        return len(self._steps) - 1

    def get_current_step_idx(self) -> int:
        return self._current_step_idx

    def get_current_state(self) -> CompleteStepState:
        return self._current_state.copy()

    def get_next_state(self) -> CompleteStepState:
        self._goto_next_state()
        return self.get_current_state()

    def get_prev_state(self) -> CompleteStepState:
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
        self._current_state.snake_alive.update(step_data.alive_states)
        self._current_state.food.update(step_data.new_food)
        self._current_state.food.difference_update(step_data.removed_food)
        self._current_state.food = set(map(lambda f: Coord(*f), self._current_state.food))
        for s_id, dir in step_data.decisions.items():
            body = self._current_state.snake_bodies[s_id]
            new_head = body[0] + dir
            body.appendleft(new_head)
            tail_dir = step_data.tail_directions[s_id]
            self._current_state.snake_ate[s_id] = new_head in step_data.removed_food
            if tail_dir != (0, 0):
                body.pop()

    def _goto_prev_state(self):
        self._current_step_idx -= 1
        if self._current_step_idx < 0:
            raise CurrentIsFirst()
        self._current_state.state_idx -= 1
        curr_step_data = self._steps[self._current_step_idx]
        self._current_state.snake_alive.update(curr_step_data.alive_states)
        # Inverse of the forward "food |= new_food; food -= removed_food". Food that
        # the same step both spawned and removed was never in the previous state, so
        # it must not be restored - hence removing new_food from both sides.
        new_food = set(curr_step_data.new_food)
        self._current_state.food.difference_update(new_food)
        self._current_state.food.update(set(curr_step_data.removed_food) - new_food)
        self._current_state.food = set(map(lambda f: Coord(*f), self._current_state.food))
        for s_id, tail_dir in curr_step_data.tail_directions.items():
            body = self._current_state.snake_bodies[s_id]
            popped_tile = body.popleft()
            if tail_dir != (0, 0):
                # body[-1] is already the tail of the state we came from, so it is
                # usable as soon as the body is non-empty. Only a snake that was a
                # single tile has to fall back to the tile we just popped.
                old_tail = body[-1] - tail_dir if len(body) > 0 else popped_tile - tail_dir
                body.append(old_tail)
        # snake_ate describes the step that produced the state we are landing on,
        # which is the step before the one being undone - and it has to be read
        # off the rewound bodies, so it cannot go in the loop above. Mirrors the
        # forward pass, which keys it on that step's decisions.
        if self._current_step_idx > 0:
            prev_step_data = self._steps[self._current_step_idx - 1]
            for s_id in prev_step_data.decisions:
                body = self._current_state.snake_bodies[s_id]
                self._current_state.snake_ate[s_id] = bool(body) and body[0] in prev_step_data.removed_food
        else:
            # The initial state is not the product of any step.
            for s_id in self._current_state.snake_ate:
                self._current_state.snake_ate[s_id] = False

    def __iter__(self):
        state_counter = 0
        self._wait_for_state(state_counter)
        while True:
            try:
                yield self.get_state(state_counter)
                state_counter += 1
            except NoMoreSteps:
                self._wait_for_state(state_counter)
            except StopIteration:
                return