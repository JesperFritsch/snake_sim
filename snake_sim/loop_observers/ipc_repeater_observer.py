
import time
import logging

from functools import wraps
from pathlib import Path
from queue import Queue, Empty
from threading import Thread, Event
from multiprocessing.connection import Connection

from snake_sim.environment.types import LoopStartData, LoopStepData, LoopStopData
from snake_sim.environment.interfaces.loop_observer_interface import ILoopObserver


log = logging.getLogger(Path(__file__).stem)

class IPCRepeaterObserver(ILoopObserver):
    def __init__(self, pipe_conn: Connection):
        self._pipe_conn = pipe_conn
        self._data_queue: Queue = Queue()
        self._stop_event: Event = Event()
        self._thread = Thread(target=self._send_worker)
        self._thread.start()

    def _check_worker(func):
        @wraps(func)
        def _check_worker_wrapper(self: "IPCRepeaterObserver", *args, **kwargs):
            if not self._thread.is_alive():
                log.warning("IPCRepeaterObserver worker thread not alive, cannot send data.")
                return
            return func(self, *args, **kwargs)
        return _check_worker_wrapper

    def _send_worker(self):
        while not self._stop_event.is_set() or not self._data_queue.empty():
            try:
                msg = self._data_queue.get(timeout=0.1)
            except Empty:
                continue
            try:
                self._pipe_conn.send(msg)
            except (BrokenPipeError, EOFError, OSError) as e:
                break
        try:
            self._pipe_conn.close()
        except:
            pass

    @_check_worker
    def notify_start(self, start_data: LoopStartData):
        self._data_queue.put(("_notify_start", start_data))

    @_check_worker
    def notify_step(self, step_data: LoopStepData):
        self._data_queue.put(("_notify_step", step_data))

    @_check_worker
    def notify_stop(self, stop_data: LoopStopData):
        self._data_queue.put(("_notify_stop", stop_data))
        self.close()

    def close(self):
        try:
            self._stop_event.set()
            self._thread.join()
        except:
            pass

