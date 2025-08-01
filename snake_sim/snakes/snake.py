import numpy as np
from collections import deque
from snake_sim.utils import coord_op, DotDict, Coord
from snake_sim.environment.interfaces.snake_interface import ISnake

DIRS = (
    (0, -1),
    (1,  0),
    (0,  1),
    (-1, 0)
)


class Snake(ISnake):
    def __init__(self):
        self.id = None
        self.start_length = None
        self.alive = True
        self.body_value = None
        self.head_value = None
        self.body_coords = deque()
        self.coord = None
        self.map = None
        self.length = self.start_length
        self.env_data = DotDict()

    def set_id(self, id: int):
        self.id = id

    def set_start_length(self, start_length: int):
        self.start_length = start_length

    def set_init_data(self, env_data):
        env_data = env_data.__dict__
        self.env_data.update(env_data)
        self.head_value = self.env_data["snake_values"][self.id]['head_value']
        self.body_value = self.env_data["snake_values"][self.id]['body_value']

    def update_env_data(self, env_data: dict):
        self.env_data.update(env_data)

    def reset(self):
        self.alive = True
        self.body_coords = deque()
        self.coord = None
        self.length = self.start_length

    def update(self, env_data):
        env_data = env_data.__dict__
        self.update_env_data(env_data)
        self.update_map(self.env_data.map)

    def is_inside(self, coord):
        x, y = coord
        return (0 <= x < self.env_data.width and 0 <= y < self.env_data.height)

    def set_start_position(self, coord: Coord):
        self.x, self.y = coord
        self.coord = tuple(coord)
        self.body_coords = deque([coord] * self.length)

    def set_new_head(self, coord):
        self.x, self.y = coord
        self.coord = coord
        if not self.is_inside(coord):
            return
        if self.map[self.y, self.x] == self.env_data.food_value:
            self.length += 1
        self.update_body(self.coord, self.body_coords, self.length)

    def _valid_tiles(self, s_map, coord, discount=None):
        """Returns a list of valid tiles from a given coord"""
        dirs = []
        for direction in DIRS:
            m_coord = coord_op(coord, direction, '+')
            x_move, y_move = m_coord
            if m_coord == discount:
                dirs.append(m_coord)
            elif not self.is_inside(m_coord):
                continue
            elif s_map[y_move, x_move] not in (self.env_data.free_value, self.env_data.food_value):
                continue
            dirs.append(m_coord)
        return dirs

    def update_body(self, new_head, body_coords: deque, length):
        body_coords.appendleft(new_head)
        old_tail = None
        for _ in range(len(body_coords) - length):
            old_tail = body_coords.pop()
        return old_tail

    def update_map(self, map: bytes):
        self.map = np.frombuffer(map, dtype=np.uint8).reshape(self.env_data.height, self.env_data.width)

    def __repr__(self) -> str:
        return f"(Class: {type(self)}, ID: {self.id}, Alive: {self.alive}, Coord: {self.coord}, Len: {self.length})"
