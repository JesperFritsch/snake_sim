import os
import datetime

from snake_env import SnakeEnv
from snakes.autoSnake import AutoSnake
from snakes.autoSnake2 import AutoSnake2
from snakes.autoSnake3 import AutoSnake3
from render.pygame_render import playback_runfile

GRID_WIDTH = 32
GRID_HEIGHT = 32
FOOD = 25

if __name__ == '__main__':
    snake_init_len = 5
    env = SnakeEnv(GRID_WIDTH, GRID_HEIGHT, FOOD)
    env.add_snake(AutoSnake('A', snake_init_len), (176, 27, 16), (176, 27, 16))
    env.add_snake(AutoSnake('B', snake_init_len), (19, 44, 209), (19, 44, 209))
    env.add_snake(AutoSnake('C', snake_init_len), (19, 212, 77), (19, 212, 77))
    env.add_snake(AutoSnake('D', snake_init_len), (199, 4, 173), (199, 4, 173))
    env.add_snake(AutoSnake('E', snake_init_len), (0, 170, 255), (0, 170, 255))
    env.add_snake(AutoSnake('F', snake_init_len), (255, 0, 0), (255, 0, 0))
    env.add_snake(AutoSnake('G', snake_init_len), (255, 162, 0), (255, 162, 0))
    env.add_snake(AutoSnake('H', snake_init_len), (250, 2, 147), (250, 2, 147))
    env.add_snake(AutoSnake('I', snake_init_len), (157, 0, 255), (157, 0, 255))
    env.add_snake(AutoSnake('J', snake_init_len), (255, 251, 0), (255, 251, 0))

    for _ in range(10):
        env.generate_run()
        env.reset()

    # playback_runfile(r"B:\pythonStuff\snake_sim\runs\grid_32x32\4_snakes_32x32_SGAOFN_ABORTED.json")

