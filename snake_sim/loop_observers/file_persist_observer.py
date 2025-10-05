
import logging
import time

from pathlib import Path
from queue import Queue, Empty
from collections import deque
from threading import Thread, Event

from snake_sim.environment.types import (
    LoopStartData,
    LoopStepData,
    LoopStopData,
)
from snake_sim.environment.interfaces.loop_observer_interface import ILoopObserver
from snake_sim.storing.run_file_handler_factory import create_run_file_handler

log = logging.getLogger(Path(__file__).stem)

DEFAULT_FILENAME = "tmp_run_file"

class FilePersistObserver(ILoopObserver):
    """ Base class for loop data consumers. Just stores all data in memory. """
    def __init__(self, store_dir: str | Path, filename: str | Path = None, file_format: str = "run_proto"):
        super().__init__()
        self._filepath: Path = Path(store_dir, filename or DEFAULT_FILENAME).with_suffix(f".{file_format}")
        self._stop_event: Event = Event()
        self._data_queue: Queue = Queue()
        self._writer_thread = Thread(target=self._write_worker)

    def _write_worker(self):
        file_handler = create_run_file_handler(str(self._filepath), new=True)
        try:
            with file_handler:
                while not self._stop_event.is_set() or not self._data_queue.empty():
                    try:
                        data = self._data_queue.get(timeout=0.1)
                        if isinstance(data, LoopStartData):
                            file_handler.write_start(data)
                        elif isinstance(data, LoopStepData):
                            file_handler.write_step(data)
                        elif isinstance(data, LoopStopData):
                            file_handler.write_stop(data)
                    except Empty:
                        pass
        finally:
            try:
                if self._filepath.stem == DEFAULT_FILENAME:
                    default_name = file_handler._default_filename()
                    new_filepath = self._filepath.with_name(default_name)
                    try:
                        self._filepath.rename(new_filepath)
                        log.info(f"Renamed temporary run file to '{new_filepath}'")
                    except Exception as e:
                        log.error(f"Failed to rename temporary run file: {e}")
            except Exception as e:
                log.error(f"Error during file renaming: {e}")

    def notify_start(self, start_data: LoopStartData):
        self._data_queue.put(start_data)
        self._writer_thread.start()

    def notify_step(self, step_data: LoopStepData):
        self._data_queue.put(step_data)

    def notify_stop(self, stop_data: LoopStopData):
        self._data_queue.put(stop_data)
        self.close()

    def close(self):
        try:
            self._stop_event.set()
            self._writer_thread.join()
        except:
            pass

    def __del__(self):
        self.close()
