
from pathlib import Path
from threading import Thread, Event

from snake_sim.storing.run_file_handler_factory import create_run_file_handler

from snake_sim.environment.interfaces.loop_observable_interface import ILoopObservable
from snake_sim.environment.types import (
    LoopStartData,
    LoopStepData,
    LoopStopData
)

class FileRepeaterObservable(ILoopObservable):

    def __init__(self, filepath: str | Path):
        self._filepath: Path = Path(filepath)
        self._worker_thread = Thread(target=self._worker, daemon=True)
        self._stop_event: Event = Event()
        super().__init__()

    def _worker(self):
        file_handler = create_run_file_handler(str(self._filepath))
        with file_handler:
            start_data = file_handler.get_start_data()
            self._notify_start(start_data)
            for step_data in file_handler.iter_steps():
                if self._stop_event.is_set():
                    break
                self._notify_step(step_data)
            stop_data = file_handler.get_stop_data()
            self._notify_stop(stop_data)

    def start(self):
        self._worker_thread.start()

    def stop(self):
        super().stop()
        self._stop_event.set()
        self._worker_thread.join()