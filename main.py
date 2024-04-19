import os
import sys
import datetime

sys.path.append(os.getcwd())

from snake_env import SnakeEnv
from snakes.autoSnake import AutoSnake
from snakes.autoSnake2 import AutoSnake2
from snakes.autoSnake3 import AutoSnake3
from snakes.autoSnake4 import AutoSnake4
from render.pygame_render import play_runfile, play_stream
from multiprocessing import Pipe, Process

GRID_WIDTH = 32
GRID_HEIGHT = 32
FOOD = 30

def start_stream_run(conn):
    snake_init_len = 5
    env = SnakeEnv(GRID_WIDTH, GRID_HEIGHT, FOOD)
    env.add_snake(AutoSnake4('A', snake_init_len, greedy=True), (176, 27, 16), (176, 27, 16))
    env.add_snake(AutoSnake4('B', snake_init_len, greedy=True), (19, 44, 209), (19, 44, 209))
    env.add_snake(AutoSnake4('C', snake_init_len, greedy=True), (19, 212, 77), (19, 212, 77))
    env.add_snake(AutoSnake4('D', snake_init_len, greedy=True), (199, 4, 173), (199, 4, 173))
    env.add_snake(AutoSnake4('E', snake_init_len, greedy=True), (0, 170, 255), (0, 170, 255))
    env.add_snake(AutoSnake4('F', snake_init_len, greedy=True), (255, 0, 0), (255, 0, 0))
    env.add_snake(AutoSnake4('G', snake_init_len, greedy=True), (255, 162, 0), (255, 162, 0))
    env.stream_run(conn,)

if __name__ == '__main__':
    snake_init_len = 5
    env = SnakeEnv(GRID_WIDTH, GRID_HEIGHT, FOOD)
    env.add_snake(AutoSnake4('A', snake_init_len, greedy=True), (176, 27, 16), (176, 27, 16))
    env.add_snake(AutoSnake4('B', snake_init_len, greedy=True), (19, 44, 209), (19, 44, 209))
    env.add_snake(AutoSnake4('C', snake_init_len, greedy=True), (19, 212, 77), (19, 212, 77))
    env.add_snake(AutoSnake4('D', snake_init_len, greedy=True), (199, 4, 173), (199, 4, 173))
    env.add_snake(AutoSnake4('E', snake_init_len, greedy=True), (0, 170, 255), (0, 170, 255))
    env.add_snake(AutoSnake4('F', snake_init_len, greedy=True), (255, 0, 0), (255, 0, 0))
    env.add_snake(AutoSnake4('G', snake_init_len, greedy=True), (255, 162, 0), (255, 162, 0))
    # env.add_snake(AutoSnake4('H', snake_init_len), (250, 2, 147), (250, 2, 147))
    # env.add_snake(AutoSnake4('I', snake_init_len), (157, 0, 255), (157, 0, 255))
    # env.add_snake(AutoSnake4('J', snake_init_len), (255, 251, 0), (255, 251, 0))
    # env.add_snake(AutoSnake4('K', snake_init_len), (52, 235, 161), (52, 235, 161))
    # env.add_snake(AutoSnake4('L', snake_init_len), (255, 255, 255), (255, 255, 255))
    # env.add_snake(AutoSnake4('M', snake_init_len), (97, 215, 255), (97, 215, 255))
    # env.add_snake(AutoSnake4('N', snake_init_len), (201, 8, 105), (201, 8, 105))
    # env.add_snake(AutoSnake4('O', snake_init_len), (107, 240, 5), (107, 240, 5))

    parent_conn, child_conn = Pipe()
    env_p = Process(target=start_stream_run, args=(child_conn,))
    render_p = Process(target=play_stream, args=(parent_conn,))
    render_p.start()
    env_p.start()
    exit_count = 0
    env_p.join()
    # parent_conn.send('stop')
    # print('stop sent')
    # for _ in range(50):
    #     env.generate_run()
    #     env.reset()

    # play_runfile(r"B:\pythonStuff\snake_sim\runs\grid_32x32\4_snakes_32x32_SGAOFN_ABORTED.json")
