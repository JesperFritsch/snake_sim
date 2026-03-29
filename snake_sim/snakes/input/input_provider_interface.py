from abc import ABC, abstractmethod

class IInputProvider(ABC):

    @abstractmethod
    def get_angle(self) -> float:
        """ Returns the angle in radians that the snake should move towards, where 0 is to the right and positive angles are counter-clockwise. """
        pass