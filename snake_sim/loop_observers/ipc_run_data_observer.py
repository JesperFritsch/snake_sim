

import logging
import functools
from pathlib import Path

from snake_sim.loop_observers.run_data_observer_interface import IRunDataObserver
from snake_sim.run_data.run_data import StepData

log = logging.getLogger(Path(__file__).stem)


class IPCRunDataObserver(IRunDataObserver):
    def __init__(self, pipe_conn):
        if not pipe_conn.__class__.__name__ in ['Connection', 'PipeConnection']:
            raise ValueError('pipe_conn must be a Connection or PipeConnection object')
        self._pipe_conn = pipe_conn
        self._disabled = False

    def handle_broken_pipe(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except (BrokenPipeError, EOFError, OSError) as e:
                log.debug('Pipe broken: %s', e)
                self._pipe_conn.close()
                self._disabled = True
        return wrapper

    def check_disabled(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if self._disabled:
                return
            return func(self, *args, **kwargs)
        return wrapper

    @check_disabled
    @handle_broken_pipe
    def notify_start(self, metadata: dict):
        self._pipe_conn.send(metadata)

    @check_disabled
    @handle_broken_pipe
    def notify_step(self, step_data: StepData):
        self._pipe_conn.send(step_data.to_dict())

    @check_disabled
    @handle_broken_pipe
    def notify_end(self, run_data):
        log.debug('Sending stopped signal')
        self._pipe_conn.send('stopped')
        self.close()

    def close(self):
        try:
            self._pipe_conn.close()
        except Exception as e:
            log.debug('Error closing pipe: %s', e)
