from collections import deque
from ..utils import exec_time

class Snake:
    def __init__(self, id: str, start_length: int):
        self.id = id.upper()
        self.start_length = start_length
        self.alive = True
        self.body_value = ord(self.id.lower())
        self.head_value = ord(self.id)
        self.body_coords = deque()
        self.coord = None
        self.env = None
        self.map = None
        self.length = self.start_length

    def reset(self):
        self.alive = True
        self.body_coords = deque()
        self.coord = None
        self.length = self.start_length

    def _init_after_bind(self):
        pass

    def bind_env(self, env):
        self.env = env
        self.height = self.env.height
        self.width = self.env.width
        self._init_after_bind()

    def in_sight(self, head_coord, coord, sight_len=2):
        h_x, h_y = head_coord
        x, y = coord
        return (h_x - sight_len) <= x <= (h_x + sight_len) and (h_y - sight_len) <= y <= (h_y + sight_len)

    def set_init_coord(self, coord):
        self.x, self.y = coord
        self.coord = coord
        self.body_coords = deque([coord] * self.length)

    def set_new_head(self, coord):
        self.x, self.y = coord
        self.coord = coord
        self.update_body(self.coord, self.body_coords, self.length)
        if self.map[self.y, self.x] == self.env.FOOD_TILE:
            self.length += 1

    def update(self):
        print("This method is not implemented")
        raise NotImplementedError

    def update_body(self, new_head, body_coords: deque, length):
        body_coords.appendleft(new_head)
        old_tail = None
        for _ in range(len(body_coords) - length):
            old_tail = body_coords.pop()
        return old_tail

    def update_map(self, map):
        self.map = map.copy()

    def kill(self):
        self.alive = False

    def __repr__(self) -> str:
        return f"(Class: {type(self)}, ID: {self.id}, Alive: {self.alive}, Coord: {self.coord}, Len: {self.length})"
