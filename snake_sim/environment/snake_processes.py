import os
import platform
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, List
from concurrent.futures import ProcessPoolExecutor, Future
from snake_sim.server.remote_snake_server import serve
from snake_sim.snakes import auto_snake


log = logging.getLogger(Path(__file__).stem)

class SnakeProcess:
    def __init__(self, id: int, target: str, module_path: str, future: Future):
        self.id = id
        self.target = target
        self.module_path = module_path
        self.future = future

    def cancel(self):
        self.check_result()
        self.future.cancel()

    def check_result(self):
        try:
            self.future.result()
        except Exception as e:
            log.error(f"Error in process with id {self.id}: {e}", exc_info=True)


class SnakeProcessPool():
    def __init__(self):
        self._executor = ProcessPoolExecutor()
        self._processes: List[SnakeProcess] = []

    def get_running_processes(self) -> List[SnakeProcess]:
        return self._processes

    def _generate_target(self, id: int) -> str:
        if platform.system() == "Windows":
            return f"localhost:{50000 + id}"
        else:
            return f"/tmp/snake_process_{id}.sock"

    def start(self, id) -> Future:
        target = self._generate_target(id)
        module_path = auto_snake.__file__
        future = self._executor.submit(serve, target=target, snake_module_file=module_path)
        self._processes.append(SnakeProcess(id, target, module_path, future))

    def shutdown(self):
        for process in self._processes:
            process.cancel()
        self._executor.shutdown(cancel_futures=True, wait=False)
        if platform.system() != "Windows":
            for process in self._processes:
                try:
                    os.remove(process.target)
                except OSError:
                    pass