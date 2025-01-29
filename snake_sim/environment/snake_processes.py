import os
import platform
import logging
import socket

from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, List
from concurrent.futures import ProcessPoolExecutor, Future
from multiprocessing import Manager
from threading import Event

from snake_sim.utils import SingletonMeta
from snake_sim.server.remote_snake_server import serve
from snake_sim.snakes import auto_snake


log = logging.getLogger(Path(__file__).stem)

class SnakeProcess:
    def __init__(self, id: int, target: str, module_path: str, future: Future, stop_event: Event = None):
        # stop_event is an event proxy from the multiprocessing.Manager, but i guess it implements the same interface as threading.Event
        self.id = id
        self.target = target
        self.module_path = module_path
        self.future = future
        self.stop_event = stop_event

    def kill(self):
        log.info(f"Stopping process with id {self.id}")
        if self.stop_event:
            self.stop_event.set()
        self.check_result()

    def check_result(self):
        try:
            self.future.result()
        except Exception as e:
            print(f"Error in process with id {self.id}: {e}", exc_info=True)


class SnakeProcessPool(metaclass=SingletonMeta):
    def __init__(self):
        self._executor = ProcessPoolExecutor(max_workers=50)
        self._processes: List[SnakeProcess] = []
        self._manager = Manager()

    def get_running_processes(self) -> List[SnakeProcess]:
        return self._processes

    def _find_free_port(self) -> int:
        """Find an available port without binding to it."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))  # Bind to a random available port
            socket_address = s.getsockname()[1]
            s.close()
            return socket_address  # Return the assigned port

    def _generate_target(self, id: int) -> str:
        if platform.system() == "Windows":
            port = self._find_free_port()
            return f"localhost:{port}"
        else:
            return f"unix:/tmp/snake_process_{id}.sock"

    def kill_snake_process(self, id: int):
        for process in self._processes:
            if process.id == id:
                process.kill()
                self._processes.remove(process)

    def start(self, id) -> Future:
        target = self._generate_target(id)
        module_path = auto_snake.__file__
        stop_event = self._manager.Event()
        future = self._executor.submit(serve, target=target, snake_module_file=module_path, stop_event=stop_event)
        self._processes.append(SnakeProcess(id, target, module_path, future, stop_event))

    def shutdown(self):
        log.debug("Shutting down processes")
        for process in self._processes:
            process.kill()
        self._executor.shutdown()
        if platform.system() != "Windows":
            for process in self._processes:
                try:
                    os.remove(process.target)
                except OSError:
                    pass