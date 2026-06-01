
import time
from typing import Callable, Tuple, List, Dict

from snake_sim.environment.interfaces.snake_updater_interface import ISnakeUpdater
from snake_sim.environment.interfaces.snake_interface import ISnake
from snake_sim.environment.types import Coord, EnvStepData, LoopDecisionData

class InprocUpdater(ISnakeUpdater):
    def __init__(self):
        super().__init__()
        pass

    def get_decisions(
        self,
        snakes: List[ISnake],
        env_step_data: EnvStepData,
        timeout_s: float | None,
        on_response: Callable[[int, int], None],
    ) -> Dict[int, Coord]:
        decisions = {}
        for snake in snakes:
            sid = snake.get_id()
            start_time = time.monotonic_ns()
            decisions[sid] = snake.update(env_step_data)
            end_time = time.monotonic_ns()
            on_response(sid, end_time - start_time)
        return decisions