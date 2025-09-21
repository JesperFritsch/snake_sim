from __future__ import annotations
import zmq
import logging
import time
from functools import wraps
from zmq.utils.monitor import recv_monitor_message
from snake_sim.environment.interfaces.snake_interface import ISnake
from snake_sim.environment.types import Coord, EnvInitData, EnvData
from snake_sim.server.shm_snake_server import Call, Return


def handle_zmq_error(func):
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except (zmq.ZMQError, zmq.Again) as e:
            raise ConnectionError
    return wrapper


class SHMProxySnake(ISnake):
    def __init__(self, target: str):
        super().__init__()
        self._log = logging.getLogger(f"{self.__class__.__name__}-{target}")
        self._target = target
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.connect(target)
        self._est_conn_timeout = 5.0
        # Make sends fail immediately if there is no connected peer
        try:
            self._socket.setsockopt(zmq.IMMEDIATE, 1)
        except Exception:
            # IMMEDIATE may not be available on very old libzmq builds
            pass
        # Do not block on close
        self._socket.setsockopt(zmq.LINGER, 0)
        self._short_timeout_set = False
        self._reader_id: int = None
        self._shm_name: str = None
        # Monitor socket and poller will be created on demand by _ensure_monitor
        self._monitor = None
        self._poller = None
        self._create_monitor()
        self._create_poller()
        self._wait_for_connection()

    @handle_zmq_error
    def zmq_msg_forwarder(func):
        @wraps(func)
        def zmq_msg_forwarder_wrapper(self: 'SHMProxySnake', *args, **kwargs):
            func(self, *args, **kwargs)
            self._send(Call(func.__name__, args[0] if args else None))
            response: Return = self._receive()
            if response.command != func.__name__:
                raise ConnectionError(f"Expected response for {func.__name__}, got {response.command}")
            return response.data
        return zmq_msg_forwarder_wrapper

    def _send(self, message: Call) -> None:
        self._socket.send(message.serialize(), flags=zmq.DONTWAIT)

    def _receive(self) -> Return:
        """Receive a message, raising ConnectionError if the connection is lost.
        """
        if self._monitor is None:
            data = self._socket.recv()
        else:
            self._wait_for_message()
            data = self._socket.recv(flags=zmq.DONTWAIT)
        return Return.deserialize(data)

    def _wait_for_connection(self):
        self._log.debug("Waiting for connection to be established...")
        start_time = time.time()
        while True:
            ready_sockets = self._poller.poll(timeout=50)
            if any(sock is self._monitor for sock, _ in ready_sockets):
                try:
                    evt = recv_monitor_message(self._monitor)
                except Exception:
                    continue
                event = evt.get('event')
                if event in (
                    zmq.EVENT_CONNECTED,
                ):
                    break
            if time.time() - start_time > self._est_conn_timeout:
                raise ConnectionError(f"Could not establish connection to server at {self._target} within {self._est_conn_timeout} seconds.")
        self._log.debug("Connection established.")


    def _check_connection_loss(self):
        ready_sockets = self._poller.poll()
        if any(sock is self._monitor for sock, _ in ready_sockets):
            try:
                evt = recv_monitor_message(self._monitor)
                self._log.debug(f"Monitor event: {evt}")
            except Exception:
                self._log.debug("Error receiving monitor message, assuming connection lost")
                return True
            event = evt.get('event')
            if event in (
                zmq.EVENT_DISCONNECTED,
                zmq.EVENT_MONITOR_STOPPED,
                zmq.EVENT_CLOSED,
            ):
                self._log.debug(f"Broken connection detected: {evt}")
                return True
        return False

    def _check_resp_ready(self):
        ready_sockets = self._poller.poll()
        return any(sock is self._socket for sock, _ in ready_sockets)

    def _create_poller(self):
        """Create a poller that watches both the monitor and the main socket.
        """
        self._poller = zmq.Poller()
        if self._monitor is not None:
            self._poller.register(self._monitor, zmq.POLLIN)
        self._poller.register(self._socket, zmq.POLLIN)
    
    def _wait_for_message(self):
        """Block until either we have data on the REQ socket or the monitor reports a disconnect.
        """
        while True:
            if self._check_connection_loss():
                raise ConnectionError
            if self._check_resp_ready():
                return
            
    def _create_monitor(self):
        """Create a monitor socket and a poller that watches both the monitor and the main socket.
        """
        try:
            self._monitor = self._socket.get_monitor_socket()
        except Exception:
            # Fallback: explicitly start monitoring and then get the monitor socket
            try:
                self._socket.monitor(None, zmq.EVENT_ALL)
                self._monitor = self._socket.get_monitor_socket()
            except Exception:
                # If monitor cannot be created, leave _monitor as None; _receive will still work
                self._monitor = None
                self._poller = zmq.Poller()
                self._poller.register(self._socket, zmq.POLLIN)
                return

    @zmq_msg_forwarder
    def set_reader_id(self, reader_id: int):
        self._reader_id = reader_id

    @zmq_msg_forwarder
    def set_shm_name(self, shm_name: str):
        self._shm_name = shm_name

    @zmq_msg_forwarder
    def kill(self):
        super().kill()

    @zmq_msg_forwarder
    def set_id(self, id: int):
        super().set_id(id)

    @zmq_msg_forwarder
    def set_start_length(self, start_length: int):
        super().set_start_length(start_length)

    @zmq_msg_forwarder
    def set_start_position(self, start_position: Coord):
        super().set_start_position(start_position)

    @zmq_msg_forwarder
    def set_init_data(self, env_init_data: EnvInitData):
        super().set_init_data(env_init_data)

    @zmq_msg_forwarder
    def shm_update(self, env_data: EnvData):
        # Define this method just to not duplicate logic by wrapping it with zmq_msg_forwarder
        pass

    def update(self, env_data: EnvData):
        # we dont send the map over zmq, because its in shared memory
        env_data.map = None
        return self.shm_update(env_data)

    def __reduce__(self):
        return (self.__class__, (self._target))

    def __del__(self):
        self._monitor.close() if self._monitor else None
        self._socket.close()
        self._context.term()
