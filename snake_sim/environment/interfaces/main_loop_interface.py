from abc import ABC, abstractmethod
from snake_sim.environment.interfaces.snake_handler_interface import ISnakeHandler
from snake_sim.environment.interfaces.snake_env_interface import ISnakeEnv
from snake_sim.environment.interfaces.loop_observer_interface import ILoopObserver


class IMainLoop(ABC):

    @abstractmethod
    def init(self, width, height):
        pass

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def stop(self):
        pass

    @abstractmethod
    def set_snake_handler(self, snake_handler: ISnakeHandler):
        pass

    @abstractmethod
    def set_max_no_food_steps(self, steps):
        pass

    @abstractmethod
    def set_max_steps(self, steps):
        pass

    @abstractmethod
    def add_observer(self, observer: ILoopObserver):
        pass

    @abstractmethod
    def set_environment(self, env: ISnakeEnv):
        pass