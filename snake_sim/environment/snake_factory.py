from snake_sim.snakes.auto_snake import AutoSnake
from snake_sim.snakes.manual_snake import ManualSnake

class SnakeFactory:
    def __init__(self):
        self.snake_types = {
            'auto': AutoSnake,
            'manual': ManualSnake
        }

    def create_snake(self, snake_type, **kwargs):
        return self.snake_types[snake_type](**kwargs)