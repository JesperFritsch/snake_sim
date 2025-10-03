

from __future__ import annotations
import logging
import functools
import time
from pathlib import Path
from threading import Thread
from collections import deque

from snake_sim.environment.interfaces.run_data_observer_interface import IRunDataObserver
from snake_sim.run_data.run_data import StepData

log = logging.getLogger(Path(__file__).stem)


class IPCRunDataObserver(IRunDataObserver):
    def __init__(self, pipe_conn):
        if not pipe_conn.__class__.__name__ in ['Connection', 'PipeConnection']:
            raise ValueError('pipe_conn must be a Connection or PipeConnection object')
        self._pipe_conn = pipe_conn
        self._disabled = False
        self._buffer: deque = deque()
        self._thread = Thread(target=self._send_worker, daemon=True)
        self._thread.start()

    def handle_broken_pipe(func):
        @functools.wraps(func)
        def wrapper(self: 'IPCRunDataObserver', *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except (BrokenPipeError, EOFError, OSError) as e:
                log.debug('Pipe broken: %s', e)
                self.close()
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
    def _send_worker(self):
        while not self._disabled:
            if not self._buffer:
                time.sleep(0.01)
                continue
            msg = self._buffer.popleft()
            self._pipe_conn.send(msg)

    def notify_start(self, metadata: dict):
        self._buffer.append(metadata)

    def notify_step(self, step_data: StepData):
        self._buffer.append(step_data.to_dict())

    def notify_stop(self, run_data):
        log.debug('Sending stopped signal')
        self._buffer.append('stopped')
        self.close()

    def close(self):
        try:
            self._pipe_conn.close()
        except Exception as e:
            log.debug('Error closing pipe: %s', e)
