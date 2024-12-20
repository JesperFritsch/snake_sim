

from multiprocessing.connection import Connection
import logging
from pathlib import Path

from snake_sim.loop_observers.run_data_observer_interface import IRunDataObserver
from snake_sim.run_data.run_data import StepData

log = logging.getLogger(Path(__file__).stem)

def handle_broken_pipe(func):
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except (BrokenPipeError, EOFError, OSError):
            log.error('Pipe broken')
            self.pipe_conn.close()
    return wrapper

class PygameRunDataObserver(IRunDataObserver):
    def __init__(self, pipe_conn: Connection):
        if not isinstance(pipe_conn, Connection):
            raise ValueError('pipe_conn must be of type Connection but is {}'.format(type(pipe_conn)))
        self.pipe_conn = pipe_conn
        self.adapter = None

    @handle_broken_pipe
    def notify_start(self, metadata: dict):
        self.pipe_conn.send(metadata)

    @handle_broken_pipe
    def notify_step(self, step_data: StepData):
        self.pipe_conn.send(step_data.to_dict())

    @handle_broken_pipe
    def notify_end(self, run_data):
        self.pipe_conn.send('stopped')
