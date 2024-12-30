

import logging
import functools
from multiprocessing.connection import PipeConnection
from pathlib import Path

from snake_sim.loop_observers.run_data_observer_interface import IRunDataObserver
from snake_sim.run_data.run_data import StepData

log = logging.getLogger(Path(__file__).stem)


class IPCRunDataObserver(IRunDataObserver):
    def __init__(self, pipe_conn: PipeConnection):
        if not isinstance(pipe_conn, PipeConnection):
            raise ValueError('pipe_conn must be of type PipeConnection but is {}'.format(type(pipe_conn)))
        self.pipe_conn = pipe_conn
        self.adapter = None

    def handle_broken_pipe(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except (BrokenPipeError, EOFError, OSError):
                log.error('Pipe broken')
                self.pipe_conn.close()
        return wrapper

    @handle_broken_pipe
    def notify_start(self, metadata: dict):
        self.pipe_conn.send(metadata)

    @handle_broken_pipe
    def notify_step(self, step_data: StepData):
        self.pipe_conn.send(step_data.to_dict())

    @handle_broken_pipe
    def notify_end(self, run_data):
        self.pipe_conn.send('stopped')
