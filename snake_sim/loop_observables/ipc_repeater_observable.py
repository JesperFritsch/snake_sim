
import logging
from pathlib import Path

from multiprocessing.connection import Connection
from threading import Thread, Event

from snake_sim.environment.types import LoopStartData, LoopStepData, LoopStopData
from snake_sim.environment.interfaces.loop_observable_interface import ILoopObservable

log = logging.getLogger(Path(__file__).stem)

class IPCRepeaterObservable(ILoopObservable):
    def __init__(self, pipe: Connection):
        super().__init__()
        self._pipe = pipe
        self._stop_event: Event = Event()
        self._thread = Thread(target=self._receiver_worker)

    def _handle_msg(self, msg):
        # example msg ("_notify_start", LoopStartData)
        n_type, n_data = msg
        getattr(self, n_type)(n_data)

    def _receiver_worker(self):
        while not self._stop_event.is_set():
            try:
                msg = self._pipe.recv()
                self._handle_msg(msg)
            except EOFError:
                break
        try:
            self._pipe.close()
        except:
            pass

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join()