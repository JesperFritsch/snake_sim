from snake_env import SnakeEnv
from snakes.dqnSnake import DqnSnake
from multiprocessing import Pipe, Process
from render.pygame_render import play_stream


def start_stream_run(conn):
    env = SnakeEnv(10, 10, 1, 1)
    env.add_snake(DqnSnake("A", 3), (255,0,0), (255,255,255))
    env.add_snake(DqnSnake("B", 3), (255,0,0), (255,255,255))
    env.stream_run(conn,)

if __name__ == '__main__':
    parent_conn, child_conn = Pipe()
    env_p = Process(target=start_stream_run, args=(child_conn,))
    render_p = Process(target=play_stream, args=(parent_conn,))
    render_p.start()
    env_p.start()
    render_p.join()
    parent_conn.send('stop')