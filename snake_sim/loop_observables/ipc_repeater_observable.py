
from multiprocessing.connection import Connection
from threading import Thread

from snake_sim.environment.types import LoopStartData, LoopStepData, LoopStopData
from snake_sim.environment.interfaces.loop_observable_interface import ILoopObservable


class IPCRepeaterObservable(ILoopObservable):
    def __init__(self, pipe: Connection):
        self._pipe = pipe
        self._current_data = None
        self._thread = Thread(target=self._receiver_worker, daemon=True)
        self._thread.start()

    def _handle_msg(self, msg):
        # example msg ("_notify_start", LoopStartData)
        n_type, n_data = msg
        self._current_data = n_data
        getattr(self, n_type)(n_data)

    def _receiver_worker(self):
        while True:
            try:
                msg = self._pipe.recv()
                self._handle_msg(msg)
            except EOFError:
                break

    def _get_start_data(self) -> LoopStartData:
        return self._current_data
    
    def _get_step_data(self) -> LoopStepData:
        return self._current_data

    def _get_stop_data(self) -> LoopStopData:
        return self._current_data
