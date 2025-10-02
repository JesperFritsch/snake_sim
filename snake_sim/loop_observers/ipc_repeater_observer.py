
import time

from collections import deque
from typing import Deque
from threading import Thread
from multiprocessing.connection import Connection

from snake_sim.environment.types import LoopStartData, LoopStepData, LoopStopData
from snake_sim.environment.interfaces.loop_observer_interface import ILoopObserver 


class IPCRepeaterObserver(ILoopObserver):
    def __init__(self, pipe_conn: Connection):
        self._pipe_conn = pipe_conn
        self._disabled = False
        self._buffer: Deque = deque()
        self._thread = Thread(target=self._send_worker, daemon=True)
        self._thread.start()

    def _send_worker(self):
        while not self._disabled:
            if not self._buffer:
                time.sleep(0.01)
                continue
            msg = self._buffer.popleft()
            try:
                self._pipe_conn.send(msg)
            except (BrokenPipeError, EOFError, OSError) as e:
                self.close()
                self._disabled = True

    def notify_start(self, start_data: LoopStartData):
        if self._disabled:
            return
        self._buffer.append(("_notify_start", start_data))

    def _notify_step(self, step_data: LoopStepData):
        if self._disabled:
            return
        self._buffer.append(("_notify_step", step_data))

    def _notify_stop(self, stop_data: LoopStopData):
        if self._disabled:
            return
        self._buffer.append(("_notify_stop", stop_data))
        self.close()

    def close(self):
        try:
            self._pipe_conn.close()
        except:
            pass
            
