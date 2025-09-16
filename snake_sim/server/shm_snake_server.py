import zmq
import platform
import pickle
import sys
import logging
import numpy as np
from typing import Optional
from threading import Event
from pathlib import Path

from snake_sim.logging_setup import setup_logging
from snake_sim.environment.interfaces.snake_interface import ISnake
from snake_sim.environment.types import Coord, EnvInitData, EnvData
from snake_sim.environment.types import SnakeConfig
from snake_sim.environment.shm_update import SharedMemoryReader


class Message:
    def __init__(self, command: str, data: any):
        self.command = command
        self.data = data

    def serialize(self) -> bytes:
        return pickle.dumps(self)
    
    @staticmethod
    def deserialize(data: bytes) -> 'Message':
        return pickle.loads(data)
    

class Call(Message):
    def __init__(self, command: str, data: any):
        super().__init__(command, data)
    
    def __str__(self):
        return f"Request(command={self.command}, data={self.data})"
    
class Return(Message):
    def __init__(self, command: str, data: any):
        super().__init__(command, data)
    
    def __str__(self):
        return f"Response(command={self.command}, data={self.data})"


class SHMSnakeServer:
    def __init__(self, shm_name: str, target: str, snake_instance: ISnake, stop_event: Optional[Event] = None):
        self._shm_name = shm_name
        self._snake_instance = snake_instance
        self.stop_event = stop_event
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(target)
        # single SharedMemoryReader for this server (one snake per server)
        self._shm_reader: SharedMemoryReader = None

    def _init_shm_reader(self, reader_id: int):
        if self._shm_reader is None:
            self._shm_reader = SharedMemoryReader(self._shm_name, reader_id)

    def _send(self, message: Return) -> None:
        self.socket.send(message.serialize())

    def _receive(self) -> Call:
        data = self.socket.recv()
        return Call.deserialize(data)
    
    def _get_reader(self, reader_id: int) -> SharedMemoryReader:
        if self._shm_reader is None:
            if reader_id != self._shm_reader._reader_id:
                raise RuntimeError(f"SHM reader ID mismatch: expected {self._shm_reader.reader_id}, got {reader_id}")
            self._init_shm_reader(reader_id)
        return self._shm_reader
    
    def serve(self):
        try:
            while not (self.stop_event and self.stop_event.is_set()):
                request = self._receive()
                snake_method = getattr(self._snake_instance, request.command, None)
                # if its not a snake method then its a command from the snake proxy to the server
                if snake_method:
                    result = snake_method(request.data)
                else:
                    result = self._handle_command(request.command, request.data)
                response = Return(request.command, result)
                self._send(response)
        except Exception as e:
            log.error(f"SHM Snake server error: {e}")
            log.debug("TRACE: ", exc_info=True)
        finally:
            self.socket.close()
            self.context.term()
            if self._shm_reader:
                self._shm_reader.close()

    def _handle_command(self, command: str, data: any) -> any:
        if command == "shm_update":
            # allow data to be either None or a dict with optional 'reader_id'
            reader_id = data["reader_id"]
            reader = self._get_reader(reader_id)
            payload = reader.read_frame()
            if payload is None:
                log.error("No payload received from shared memory reader")
                return None

            env_data: EnvData = data["env_data"]
            env_data.map = payload
            # Call the snake update method with EnvData and return the result
            try:
                result = self._snake_instance.update(env_data)
                return result
            except Exception as e:
                logging.exception("Snake update failed")
                return None
        else:
            raise ValueError(f"Unknown command {command}")


def import_snake_module(snake_module_file):
    if snake_module_file:
        snake_file_path = Path(snake_module_file)
        sys.path.append(str(snake_file_path.parent))
        import importlib.util
        spec = importlib.util.spec_from_file_location("snake_module", snake_module_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    return None


def serve(
    target: str,
    shm_name: str,
    snake_module_file=None, 
    snake_config: SnakeConfig=None, 
    stop_event: Optional[Event] = None,
    log_level=logging.INFO
):
    if platform.system() == "Windows":
        setup_logging(log_level)
    
    global log
    log = logging.getLogger(f"{target}")

    try:
        if not bool(snake_module_file) ^ bool(snake_config):
            raise ValueError("Either snake_module_file or snake_config must be provided, but not both and not neither")

        if snake_module_file:
            snake_module = import_snake_module(snake_module_file)
            snake_instance = snake_module.MySnake()

        elif snake_config:
            # Only import here to avoid letting snake_module_file see our environment and code.
            from snake_sim.environment.snake_factory import SnakeFactory, SnakeProcType
            from snake_sim.snakes.strategies.utils import apply_strategies
            factory = SnakeFactory()
            _, snake_instance = factory.create_snake(
                SnakeProcType.LOCAL,
                snake_config
            )
            apply_strategies(snake_instance, snake_config)

        server = SHMSnakeServer(shm_name, target, snake_instance, stop_event)
        log.info(f"Starting SHM snake server on {target}")
        server.serve()

    except Exception as e:
        log.error(e)
        log.debug("TRACE: ", exc_info=True)

