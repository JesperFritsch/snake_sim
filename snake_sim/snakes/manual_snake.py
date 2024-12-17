

import random

from snake_sim.snakes.snake import Snake
from snake_sim.utils import coord_op


class ManualSnake(Snake):
    def __init__(self, id: str, start_length: int, help=0):
        super().__init__(id, start_length)
        self.help = 0
        self.last_direction_coord = (0, 0)
        self.next_direction = (0, -1)

    def update(self, env_data):
        super().update(env_data)
        current_direction = coord_op(self.coord, self.body_coords[1], "-")
        new_direction = self.next_direction
        # print(f"Current direction: {current_direction}")
        # print(f"New direction: {new_direction}")
        print(f"next_direction: {self.next_direction}")
        if self.help >= 1:
            next_tile = coord_op(self.coord, new_direction, "+")
            if not next_tile in self._valid_tiles(self.map, self.coord):
                valids = self._valid_tiles(self.map, self.coord)
                tile_straight = coord_op(self.coord, current_direction, "+")
                if tile_straight in valids:
                    new_direction = current_direction
                elif valids:
                    next_tile = random.choice(valids)
                    new_direction = coord_op(next_tile, self.coord, "-")
        self.last_direction_coord = new_direction
        self.set_new_head(coord_op(self.coord, new_direction, "+"))
        return new_direction

    def set_direction_explicit(self, direction: str):
        direction = direction.upper()
        if direction in ("UP", "DOWN", "LEFT", "RIGHT"):
            if direction == "UP":
                self.next_direction = (0, -1)
            elif direction == "DOWN":
                self.next_direction = (0, 1)
            elif direction == "LEFT":
                self.next_direction = (-1, 0)
            else:
                self.next_direction = (1, 0)
        else:
            raise ValueError("Invalid direction choice")
        print(f"next_direction: {self.next_direction}")


    def set_direction_implicit(self, direction: str):
        current_direction = coord_op(self.coord, self.body_coords[1], "-")
        direction = direction.upper()
        if direction == "LEFT":
            if current_direction == (1, 0):
                new_direction = (0, -1)
            elif current_direction == (0, 1):
                new_direction = (1, 0)
            elif current_direction == (0, -1):
                new_direction = (-1, 0)
            else:
                new_direction = (0, 1)
        elif direction == "RIGHT":
            if current_direction == (1, 0):
                new_direction = (0, 1)
            elif current_direction == (0, 1):
                new_direction = (-1, 0)
            elif current_direction == (0, -1):
                new_direction = (1, 0)
            else:
                new_direction = (0, -1)
        else:
            raise ValueError("Invalid direction choice")
        self.next_direction = new_direction

    def set_direction(self, direction: str):
        """ Takes a direction command and sets the next direction to move in
            A command looks like '<explicit/implicit>_direction' ex: 'explicit_up' or 'implicit_left'
            implicit can only be left or right
        """
        control_type, direction = direction.split("_")
        if control_type == "explicit":
            self.set_direction_explicit(direction)
        elif control_type == "implicit":
            self.set_direction_implicit(direction)
