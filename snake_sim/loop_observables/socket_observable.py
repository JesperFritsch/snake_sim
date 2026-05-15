# services/runner/runner/socket_observable.py
import logging
import socket
import struct
from pathlib import Path
from threading import Thread, Event
from typing import Optional

from snake_sim.environment.interfaces.loop_observable_interface import ILoopObservable
from snake_sim.environment.types import (
    LoopStartData,
    LoopStepData,
    LoopStopData,
    EnvMetaData,
    Coord,
)

from snake_sim.protobuf import simrun_pb2

log = logging.getLogger(__name__)

MSG_START = 1
MSG_STEP = 2
MSG_STOP = 3


def _coord(c) -> Coord:
    return Coord(x=c.x, y=c.y)


def _deserialize_start(payload: bytes) -> LoopStartData:
    proto = simrun_pb2.LoopStartData.FromString(payload)
    md = proto.env_meta_data
    # NOTE: base_map is bytes here; if your LoopStartData consumers need a numpy array
    # you can reshape with np.frombuffer(md.base_map, dtype=md.base_map_dtype).reshape(md.height, md.width)
    return LoopStartData(
        env_meta_data=EnvMetaData(
            height=md.height,
            width=md.width,
            free_value=md.free_value,
            blocked_value=md.blocked_value,
            food_value=md.food_value,
            snake_tags={int(k): v for k, v in md.snake_tags.items()},
            snake_values={
                k: {"head_value": v.head_value, "body_value": v.body_value}
                for k, v in md.snake_values.items()
            },
            start_positions={k: _coord(v) for k, v in md.start_positions.items()},
            base_map=md.base_map,
            base_map_dtype=md.base_map_dtype,
        )
    )


def _deserialize_step(payload: bytes) -> LoopStepData:
    proto = simrun_pb2.LoopStepData.FromString(payload)
    return LoopStepData(
        step=proto.step,
        total_time=proto.total_time,
        alive_states=dict(proto.alive_states),
        snake_times=dict(proto.snake_times),
        decisions={k: _coord(v) for k, v in proto.decisions.items()},
        tail_directions={k: _coord(v) for k, v in proto.tail_directions.items()},
        snake_grew=dict(proto.snake_grew),
        lengths=dict(proto.lengths),
        new_food=[_coord(c) for c in proto.new_food],
        removed_food=[_coord(c) for c in proto.removed_food],
    )


def _deserialize_stop(payload: bytes) -> LoopStopData:
    proto = simrun_pb2.LoopStopData.FromString(payload)
    return LoopStopData(final_step=proto.final_step)


def _read_exact(sock: socket.socket, n: int) -> Optional[bytes]:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


class SocketObservable(ILoopObservable):
    """
    Binds a TCP port and waits for exactly one sender (the sim's SocketObserver).
    Forwards every received start/step/stop event to attached observers.
    """

    def __init__(self, bind_host: str = "0.0.0.0", bind_port: int = 0, accept_timeout_ms: int = 30.0):
        super().__init__()
        self._listener = socket.socket()
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener.bind((bind_host, bind_port))
        self._listener.listen(1)
        self._listener.settimeout(accept_timeout_ms)
        self._bound_port = self._listener.getsockname()[1]
        self._stop_event: Event = Event()
        self._thread: Optional[Thread] = None
        self._conn: Optional[socket.socket] = None
        self._latest_start: Optional[LoopStartData] = None
        self._latest_stop: Optional[LoopStopData] = None

    @property
    def port(self) -> int:
        return self._bound_port

    def _get_start_data(self) -> LoopStartData:
        return self._latest_start

    def _get_step_data(self) -> LoopStepData:
        raise NotImplementedError  # we forward directly, never queried

    def _get_stop_data(self) -> LoopStopData:
        return self._latest_stop

    def start(self):
        self._thread = Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self):
        try:
            self._conn, _ = self._listener.accept()
        except socket.timeout:
            log.error("SocketObservable: no sender connected within timeout")
            return
        finally:
            self._listener.close()

        self._conn.settimeout(None)
        try:
            while not self._stop_event.is_set():
                header = _read_exact(self._conn, 5)
                if header is None:
                    log.info("SocketObservable: sender closed the connection")
                    return
                msg_type, length = struct.unpack(">BI", header)
                payload = _read_exact(self._conn, length)
                if payload is None:
                    log.warning("SocketObservable: truncated payload")
                    return

                if msg_type == MSG_START:
                    self._latest_start = _deserialize_start(payload)
                    self._notify_start()
                elif msg_type == MSG_STEP:
                    step = _deserialize_step(payload)
                    for observer in self._observers:
                        observer.notify_step(step)
                elif msg_type == MSG_STOP:
                    self._latest_stop = _deserialize_stop(payload)
                    self._notify_stop()
                    return
                else:
                    log.warning("SocketObservable: unknown msg type %d", msg_type)
        finally:
            try:
                self._conn.close()
            except OSError:
                pass

    def stop(self):
        self._stop_event.set()
        try:
            if self._conn:
                self._conn.shutdown(socket.SHUT_RDWR)
                self._conn.close()
        except OSError:
            pass
        try:
            self._listener.close()
        except OSError:
            pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

    def close(self):
        self.stop()
        super().close()