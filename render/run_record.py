from collections.abc import Iterable
import json
from collections import deque


class StepData:
    def __init__(self, food: list, step: int) -> None:
        self.snakes = []
        self.food = food
        self.step = step

    def add_snake_data(self, snake_coords: list, head_dir: tuple, tail_dir: tuple, snake_id: str):
        self.snakes.append({
            'snake_id': snake_id,
            'coords': snake_coords,
            'head_dir': head_dir,
            'tail_dir': tail_dir
        })

    def to_dict(self):
        return {
            'snakes': self.snakes,
            'food': self.food,
            'step': self.step
        }


class RunData:
    def __init__(self, width: int, height: int, snake_data: list) -> None:
        self.width = width
        self.height = height
        self.snake_data = snake_data
        self.steps = {}

    def add_step(self, step: int, state: StepData):
        self.steps[step] = state

    def to_dict(self):
        return {
            'width': self.width,
            'height': self.height,
            'snake_data': self.snake_data,
            'steps': {k: v.to_dict() for k, v in self.steps.items()}
        }
