from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict
from snake_sim.environment.types import Coord, EnvInitData, EnvData
from snake_sim.environment.interfaces.food_handler_interface import IFoodHandler


class ISnakeEnv(ABC):
    @abstractmethod
    def get_env_data(self, id: Optional[int]=None) -> EnvData:
        pass

    @abstractmethod
    def get_init_data(self) -> EnvInitData:
        pass

    @abstractmethod
    def move_snake(self, id: int, direction: Coord) -> Tuple[bool, bool]: #(alive, ate)
        pass

    @abstractmethod
    def load_map(self, map_img_path: str):
        pass

    @abstractmethod
    def add_snake(self, id: int, start_position: Optional[Coord]=None, start_length: int=3):
        pass

    @abstractmethod
    def set_food_handler(self, food_handler: IFoodHandler):
        pass

    @abstractmethod
    def resize(self, height: int, width: int):
        pass

    @abstractmethod
    def update_food(self):
        pass

    @abstractmethod
    def steps_since_any_ate(self):
        pass

    @abstractmethod
    def get_food(self):
        pass

    @abstractmethod
    def get_head_positions(self) -> Dict[int, Coord]:
        pass
