import os
import platform
import logging
import socket
from pathlib import Path
from typing import Dict, List
from multiprocessing import Process, Manager

from snake_sim.utils import SingletonMeta, rand_str
from snake_sim.server.remote_snake_server import serve
from snake_sim.snakes import auto_snake

log = logging.getLogger(Path(__file__).stem)

class SnakeProcess:
    def __init__(self, id: int, target: str, module_path: str, process: Process, stop_event=None):
        self.id = id
        self.target = target
        self.module_path = module_path
        self.process = process
        self.stop_event = stop_event

    def is_running(self) -> bool:
        return self.process.is_alive()

    def kill(self):
        if self.process.is_alive():
            log.info(f"Stopping process with id {self.id}")
            if platform.system() == "Windows":
                self.process.terminate()
                self.process.join()
            else:
                if self.stop_event:
                    try:
                        self.stop_event.set()
                    except Exception as e:
                        log.error(f"Error setting stop event: {e}")
                    self.process.join()
                else:
                    log.warning("No stop event found for process, killing process with SIGKILL")
                    self.process.kill()
        if self.target.startswith("unix:"):
            try:
                filename = self.target.split(':')[1]
                if Path(filename).exists():
                    os.remove(filename)
            except OSError as e:
                log.error(f"Could not remove socket file {self.target}: {e}")


class ProcessPool(metaclass=SingletonMeta):
    def __init__(self):
        self._processes: Dict[int, SnakeProcess] = {}
        self._manager = Manager()

    def get_running_processes(self) -> List[SnakeProcess]:
        return self._processes.values()

    def _find_free_port(self) -> int:
        """Find an available port without binding to it."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))  # Bind to a random available port
            socket_address = s.getsockname()[1]
            s.close()
            return socket_address  # Return the assigned port

    def _generate_target(self) -> str:
        if platform.system() == "Windows":
            port = self._find_free_port()
            return f"localhost:{port}"
        else:
            sock_file = f"/tmp/snake_process_{rand_str(8)}.sock"
            while Path(sock_file).exists():
                sock_file = f"/tmp/snake_process_{rand_str(8)}.sock"
            return f"unix:{sock_file}"

    def is_running(self, id: int) -> bool:
        return id in self._processes and self._processes[id].is_running()

    def kill_snake_process(self, id: int):
        if id in self._processes:
            self._processes[id].kill()
            del self._processes[id]

    def start(self, id):
        target = self._generate_target()
        module_path = auto_snake.__file__
        stop_event = self._manager.Event()
        process = Process(target=serve, args=(target, module_path, stop_event))
        process.start()
        self._processes[id] = SnakeProcess(id, target, module_path, process, stop_event)

    def shutdown(self):
        log.debug("Shutting down processes")
        processes = self._processes.copy()
        for process in processes.keys():
            self.kill_snake_process(process)
        for process in processes.values():
            process.process.join()
        self._manager.shutdown()