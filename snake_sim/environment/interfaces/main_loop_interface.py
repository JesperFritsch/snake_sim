from abc import ABC, abstractmethod
from snake_sim.environment.interfaces.snake_handler_interface import ISnakeHandler
from snake_sim.environment.interfaces.snake_env_interface import ISnakeEnv
from snake_sim.environment.interfaces.loop_observer_interface import ILoopObserver
from snake_sim.environment.interfaces.loop_observable_interface import ILoopObservable


class IMainLoop(ABC, ILoopObservable):

    def __init__(self, *args, **kwargs):
        super().__init__()

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
    def set_environment(self, env: ISnakeEnv):
        pass
