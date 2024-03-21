import os
from snake_env import SnakeEnv
from snakes.autoSnake import AutoSnake
from render.pygame_render import playback_runfile

GRID_WIDTH = 20
GRID_HEIGHT = 20

if __name__ == '__main__':
    runs_dir = os.path.abspath(os.path.join(os.getcwd(), 'runs'))
    run_len = len(os.listdir(runs_dir))
    DEFAULT_FILENAME = os.path.join(runs_dir, f'snake_run{3}.json')
    snake_init_len = 5
    env = SnakeEnv(GRID_WIDTH, GRID_HEIGHT, 25)
    env.add_snake(AutoSnake('A', snake_init_len), (176, 27, 16), (125, 19, 11))
    env.add_snake(AutoSnake('B', snake_init_len), (19, 44, 209), (8, 23, 120))
    env.add_snake(AutoSnake('C', snake_init_len), (19, 212, 77), (10, 140, 49))
    env.add_snake(AutoSnake('D', snake_init_len), (128, 3, 111), (199, 4, 173))

    # env.generate_run(DEFAULT_FILENAME)
    playback_runfile(DEFAULT_FILENAME)

