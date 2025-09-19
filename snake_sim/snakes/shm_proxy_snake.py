
from __future__ import annotations
import zmq
import logging
from snake_sim.environment.interfaces.snake_interface import ISnake
from snake_sim.environment.types import Coord, EnvInitData, EnvData
from snake_sim.server.shm_snake_server import Call, Return


def handle_connection_loss(func):
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except zmq.ZMQError as e:
            raise ConnectionError
    return wrapper


class SHMProxySnake(ISnake):
    def __init__(self, target: str):
        super().__init__()
        self._log = logging.getLogger(f"{self.__class__.__name__}-{target}")
        self._target = target
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.HEARTBEAT_IVL, 10)  # 10 ms heartbeat interval
        self._socket.setsockopt(zmq.HEARTBEAT_TIMEOUT, 100)  # 100 ms heartbeat timeout
        self._socket.connect(target)
        self._short_timeout_set = False
        self._reader_id: int = None
        self._shm_name: str = None

    def zmq_msg_forwarder(func):
        def wrapper(self: 'SHMProxySnake', *args, **kwargs):
            func(self, *args, **kwargs)
            self._send(Call(func.__name__, args[0] if args else None))
            response: Return = self._receive()
            if response.command != func.__name__:
                raise ConnectionError(f"Expected response for {func.__name__}, got {response.command}")
            return response.data
        return wrapper

    def _set_short_timeout(self):
        if not self._short_timeout_set:
            self._socket.setsockopt(zmq.RCVTIMEO, 500)
            self._socket.setsockopt(zmq.SNDTIMEO, 10)
            self._short_timeout_set = True

    def _send(self, message: Call) -> None:
        self._socket.send(message.serialize())

    def _receive(self) -> Return:
        data = self._socket.recv()
        return Return.deserialize(data)

    @handle_connection_loss
    @zmq_msg_forwarder
    def set_reader_id(self, reader_id: int):
        self._reader_id = reader_id

    @handle_connection_loss
    @zmq_msg_forwarder
    def set_shm_name(self, shm_name: str):
        self._shm_name = shm_name

    @handle_connection_loss
    @zmq_msg_forwarder
    def kill(self):
        super().kill()

    @handle_connection_loss
    @zmq_msg_forwarder
    def set_id(self, id: int):
        super().set_id(id)

    @handle_connection_loss
    @zmq_msg_forwarder
    def set_start_length(self, start_length: int):
        super().set_start_length(start_length)

    @handle_connection_loss
    @zmq_msg_forwarder
    def set_start_position(self, start_position: Coord):
        super().set_start_position(start_position)

    @handle_connection_loss
    @zmq_msg_forwarder
    def set_init_data(self, env_init_data: EnvInitData):
        super().set_init_data(env_init_data)

    @handle_connection_loss
    @zmq_msg_forwarder
    def shm_update(self, env_data: EnvData):
        # Define this method just to not duplicate logic.
        pass

    def update(self, env_data: EnvData):
        # we dont send the map over zmq, because its in shared memory
        self._set_short_timeout()
        env_data.map = None
        return self.shm_update(env_data)

    def __reduce__(self):
        return (self.__class__, (self._target))

    def __del__(self):
        try:
            self.kill()
        except Exception:
            pass
