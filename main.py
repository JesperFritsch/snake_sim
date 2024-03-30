import os
import datetime

from snake_env import SnakeEnv
from snakes.autoSnake import AutoSnake
from snakes.autoSnake2 import AutoSnake2
from snakes.autoSnake3 import AutoSnake3
from render.pygame_render import playback_runfile

GRID_WIDTH = 32
GRID_HEIGHT = 32
FOOD = 20

if __name__ == '__main__':
    snake_init_len = 5
    env = SnakeEnv(GRID_WIDTH, GRID_HEIGHT, FOOD)
    env.add_snake(AutoSnake3('A', snake_init_len), (176, 27, 16), (125, 19, 11))
    env.add_snake(AutoSnake3('B', snake_init_len), (19, 44, 209), (8, 23, 120))
    env.add_snake(AutoSnake3('C', snake_init_len), (19, 212, 77), (10, 140, 49))
    env.add_snake(AutoSnake3('D', snake_init_len), (128, 3, 111), (199, 4, 173))

    for _ in range(10):
        env.generate_run()
        env.reset()

    # playback_runfile(r"B:\pythonStuff\snake_sim\runs\grid_32x32\4_snakes_32x32_SGAOFN_ABORTED.json")

