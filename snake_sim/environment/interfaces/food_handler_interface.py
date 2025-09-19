from abc import ABC, abstractmethod


class IFoodHandler(ABC):

    @abstractmethod
    def update(self, s_map):
        pass

    @abstractmethod
    def resize(self, width, height):
        pass

    @abstractmethod
    def clear(self):
        pass

    @abstractmethod
    def remove(self, coord, s_map):
        pass

    @abstractmethod
    def add_new(self, coord):
        pass

    @abstractmethod
    def get_food(self, only_new=False):
        pass