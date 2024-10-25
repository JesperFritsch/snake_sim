


from snake_sim.snakes.snake import Snake
from snake_sim.utils import coord_op
class ManualSnake(Snake):
    def __init__(self, id: str, start_length: int):
        super().__init__(id, start_length)
        self.direction_choice = None
        self.last_direction_coord = None

    def _init_after_bind(self):
        self.update_map(self.env.map)
        valid_tiles = self._valid_tiles(self.map, self.coord)
        self.last_direction_coord = coord_op(valid_tiles[0], self.coord, "-")

    def update(self):
        self.update_map(self.env.map)
        current_direction = coord_op(self.coord, self.body_coords[1], "-")
        new_direction = None
        if self.direction_choice == "LEFT":
            if current_direction == (1, 0):
                new_direction = (0, -1)
            elif current_direction == (0, 1):
                new_direction = (1, 0)
            elif current_direction == (0, -1):
                new_direction = (-1, 0)
            else:
                new_direction = (0, 1)
        elif self.direction_choice == "RIGHT":
            if current_direction == (1, 0):
                new_direction = (0, 1)
            elif current_direction == (0, 1):
                new_direction = (-1, 0)
            elif current_direction == (0, -1):
                new_direction = (1, 0)
            else:
                new_direction = (0, -1)
        elif self.direction_choice is None:
            new_direction = self.last_direction_coord
        else:
            raise ValueError("Invalid direction choice")
        self.last_direction_coord = new_direction
        self.direction_choice = None
        return new_direction

    def set_direction_choice(self, direction_choice: str):
        self.direction_choice = direction_choice.upper()

    def get_direction_choice(self):
        return self.direction_choice
