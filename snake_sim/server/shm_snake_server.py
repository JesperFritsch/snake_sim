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
from snake_sim.environment.types import EnvData, SnakeConfig, SnakeProcType
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
    def __init__(self, target: str, snake_instance: ISnake, stop_event: Event):
        self._snake_instance = snake_instance
        self.stop_event = stop_event
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.setsockopt(zmq.RCVTIMEO, 50)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.bind(target)
        # single SharedMemoryReader for this server (one snake per server)
        self._shm_reader: SharedMemoryReader = None
        self._shm_name: str = None
        self._shm_reader_id: int = None

    def _init_shm_reader(self):
        if self._shm_name is None or self._shm_reader_id is None:
            return # not enough info to init
        if self._shm_reader is None:
            self._shm_reader = SharedMemoryReader(self._shm_name, self._shm_reader_id)

    def _send(self, message: Return) -> None:
        self.socket.send(message.serialize())

    def _receive(self) -> Call:
        data = self.socket.recv()
        rec = Call.deserialize(data)
        return rec

    def _get_reader(self) -> SharedMemoryReader:
        if self._shm_reader is None:
            raise ValueError("SharedMemoryReader not initialized. Call set_reader_id and set_shm_name first.")
        return self._shm_reader

    def set_reader_id(self, reader_id: int):
        self._shm_reader_id = reader_id
        self._init_shm_reader()

    def set_shm_name(self, shm_name: str):
        self._shm_name = shm_name
        self._init_shm_reader()

    def shm_update(self, env_data: EnvData):
        reader = self._get_reader()
        payload = reader.read_frame()
        if payload is None:
            log.error("No payload received from shared memory reader")
            return None

        if not isinstance(env_data, EnvData):
            raise ValueError("Expected EnvData as data for shm_update command")
        # Update the EnvData with the new map from shared memory
        env_data.map = payload
        # Call the snake update method with EnvData and return the result
        try:
            result = self._snake_instance.update(env_data)
            return result
        except Exception as e:
            logging.exception("Snake update failed")
            return None

    def serve(self):
        try:
            while not self.stop_event.is_set():
                try:
                    request = self._receive()
                except zmq.Again:
                    continue
                # if its not a snake method then its a command from the snake proxy to the server
                if hasattr(self._snake_instance, request.command):
                    snake_method = getattr(self._snake_instance, request.command)
                    result = snake_method(request.data)
                elif hasattr(self, request.command):
                    server_method = getattr(self, request.command)
                    result = server_method(request.data)
                else:
                    raise ValueError(f"Unknown command: {request.command}")
                response = Return(request.command, result)
                self._send(response)
            log.debug("Stop event set, shutting down server")
        except (BrokenPipeError, EOFError, ConnectionError):
            log.debug("Client disconnected, shutting down server")
        except KeyboardInterrupt:
            pass
        except Exception as e:
            log.error(f"SHM Snake server error: {e}")
            log.debug("TRACE: ", exc_info=True)
        finally:
            self.socket.close()
            self.context.term()
            if self._shm_reader:
                self._shm_reader.close()


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
    snake_module_file=None,
    snake_config: SnakeConfig=None,
    stop_event: Optional[Event] = None,
    log_level=logging.INFO
):
    if platform.system() == "Windows":
        setup_logging(log_level)
    global log
    log = logging.getLogger(f"{Path(__file__).stem}-{target}")

    try:
        if not bool(snake_module_file) ^ bool(snake_config):
            raise ValueError("Either snake_module_file or snake_config must be provided, but not both and not neither")

        if snake_module_file:
            snake_module = import_snake_module(snake_module_file)
            snake_instance = snake_module.MySnake()

        elif snake_config:
            # Only import here to avoid letting snake_module_file see our environment and code.
            from snake_sim.environment.snake_factory import SnakeFactory
            factory = SnakeFactory()
            snake_instance = factory.create_snake(
                snake_config=snake_config
            )
        server = SHMSnakeServer(target, snake_instance, stop_event)
        log.info(f"Starting SHM snake server on {target}")
        server.serve()

    except Exception as e:
        log.error(e)
        log.debug("TRACE: ", exc_info=True)



