import os
import sys
import datetime
import argparse

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
FOOD = 15

snake_configs = [
    {
        'snake':{
            'id': 'A',
            'start_length': 5,
            'greedy': True
        },
        'env':{
            'h_color': (176, 27, 16),
            'b_color': (125, 19, 11)
        }
    },
    {
        'snake':{
            'id': 'B',
            'start_length': 5,
            'greedy': True
        },
        'env':{
            'h_color': (19, 44, 209),
            'b_color': (12, 26, 117)
        }
    },
    {
        'snake':{
            'id': 'C',
            'start_length': 5,
            'greedy': True
        },
        'env':{
            'h_color': (19, 212, 77),
            'b_color': (12, 117, 43)
        }
    },
    {
        'snake':{
            'id': 'D',
            'start_length': 5,
            'greedy': True
        },
        'env':{
            'h_color': (199, 4, 173),
            'b_color': (139, 2, 121)
        }
    },
    {
        'snake':{
            'id': 'E',
            'start_length': 5,
            'greedy': True
        },
        'env':{
            'h_color': (0, 170, 255),
            'b_color': (0, 119, 179)
        }
    },
    {
        'snake':{
            'id': 'F',
            'start_length': 5,
            'greedy': True
        },
        'env':{
            'h_color': (255, 0, 0),
            'b_color': (179, 0, 0)
        }
    },
    {
        'snake':{
            'id': 'G',
            'start_length': 5,
            'greedy': True
        },
        'env':{
            'h_color': (255, 162, 0),
            'b_color': (179, 114, 0)
        }
    }
]

def start_stream_run(conn):
    env = SnakeEnv(GRID_WIDTH, GRID_HEIGHT, FOOD)
    snake_count = 4
    count = 0
    for config in snake_configs:
        count += 1
        env.add_snake(AutoSnake4(**config['snake']), **config['env'])
        if count == snake_count:
            break
    env.stream_run(conn,)

if __name__ == '__main__':
    snake_init_len = 5
    env = SnakeEnv(GRID_WIDTH, GRID_HEIGHT, FOOD)
    snake_count = 7
    count = 0
    for config in snake_configs:
        count += 1
        env.add_snake(AutoSnake4(**config['snake']), **config['env'])
        if count == snake_count:
            break
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
