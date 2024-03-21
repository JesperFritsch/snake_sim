import os
from snake_env import SnakeEnv
from snakes.autoSnake import AutoSnake

TILE_SIZE_PX = 40
GRID_WIDTH = 15
GRID_HEIGHT = 15
TILE_SIZE = 3

if __name__ == '__main__':
    DEFAULT_FILENAME = os.path.abspath(os.path.join(os.getcwd(), 'runs', 'snake_run.json'))
    window_width = GRID_WIDTH * TILE_SIZE_PX
    window_height = GRID_HEIGHT * TILE_SIZE_PX
    snake_len = 5
    env = SnakeEnv(GRID_WIDTH, GRID_HEIGHT, 15)
    env.add_snake(AutoSnake('A', snake_len), (176, 27, 16), (125, 19, 11))
    # env.add_snake(AutoSnake('B', snake_len), (19, 44, 209), (8, 23, 120))
    # env.add_snake(AutoSnake('C', snake_len), (19, 212, 77), (10, 140, 49))
    # env.add_snake(AutoSnake('D', snake_len), (128, 3, 111), (199, 4, 173))

    env.generate_run(DEFAULT_FILENAME)


