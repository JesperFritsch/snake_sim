
from typing import Callable, Tuple, List, Dict

from snake_sim.environment.interfaces.snake_updater_interface import ISnakeUpdater
from snake_sim.environment.interfaces.snake_interface import ISnake
from snake_sim.environment.types import Coord, EnvStepData

class InprocUpdater(ISnakeUpdater):
    def __init__(self):
        super().__init__()
        pass

    def get_decisions(
        self,
        snakes: List[ISnake],
        env_step_data: EnvStepData,
        timeout_s: float | None,
        on_response: Callable[[int], None],
    ) -> Dict[int, Coord]:
        decisions = {}
        for snake in snakes:
            sid = snake.get_id()
            decisions[sid] = snake.update(env_step_data)
            on_response(sid)
        return decisions