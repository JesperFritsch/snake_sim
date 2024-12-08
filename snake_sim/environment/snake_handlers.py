from abc import ABC, abstractmethod

from snake_sim.snakes.snake import ISnake

class ISnakeHandler(ABC):
    @abstractmethod
    def get_decision(self, id, env_data: dict) -> tuple:
        pass

    @abstractmethod
    def add_snake(self, snake: ISnake):
        pass

    @abstractmethod
    def get_snake_order(self):
        pass


class SnakeHandler(ISnakeHandler):
    def __init__(self):
        self.snakes = {}

    def get_decision(self, id, env_data: dict) -> tuple:
        pass

    def add_snake(self, snake: ISnake, h_color: tuple, b_color: tuple):
        pass

    def get_snake_order(self):
        pass