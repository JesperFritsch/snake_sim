from snake_env import SnakeEnv
from snakes.dqnSnake import DqnSnake
from multiprocessing import Pipe, Process
from render.pygame_render import play_stream

GRID_WIDTH = 10
GRID_HEIGHT = 10

agent_in_height = GRID_HEIGHT * 2
agent_in_width = GRID_WIDTH * 2


def start_stream_run(conn):
    env = SnakeEnv(GRID_WIDTH, GRID_HEIGHT, 5)
    env.add_snake(DqnSnake("A", 3, agent_in_height, agent_in_width), (255,0,0), (255,255,255))
    # env.add_snake(DqnSnake("B", 3), (255,0,0), (255,255,255))
    env.stream_run(conn,)

def start_traing_run():
    env = SnakeEnv(GRID_WIDTH, GRID_HEIGHT, 5)
    env.add_snake(DqnSnake("A", 2, agent_in_height, agent_in_width, training=True), (255,0,0), (255,255,255))
    env.ml_training_run(30000)

if __name__ == '__main__':
    if 0:
        start_traing_run()
    else:
        parent_conn, child_conn = Pipe()
        env_p = Process(target=start_stream_run, args=(child_conn,))
        render_p = Process(target=play_stream, args=(parent_conn,))
        render_p.start()
        env_p.start()
        render_p.join()
        parent_conn.send('stop')