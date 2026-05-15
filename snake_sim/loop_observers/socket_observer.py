# snake_sim/loop_observers/socket_observer.py
import logging
import socket
import struct
from pathlib import Path
from queue import Queue, Empty
from threading import Thread, Event

from snake_sim.environment.types import LoopStartData, LoopStepData, LoopStopData
from snake_sim.environment.interfaces.loop_observer_interface import ILoopObserver
from snake_sim.environment.types import EnvMetaData, Coord
from snake_sim.protobuf import simrun_pb2  # generated from your proto

log = logging.getLogger(Path(__file__).stem)

# message type tags — single byte prefix, before the length
MSG_START = 1
MSG_STEP = 2
MSG_STOP = 3


def _coord(c: Coord) -> simrun_pb2.Coord:
    return simrun_pb2.Coord(x=c.x, y=c.y)


def _serialize_start(data: LoopStartData) -> bytes:
    md = data.env_meta_data
    proto = simrun_pb2.LoopStartData(
        env_meta_data=simrun_pb2.EnvMetaData(
            height=md.height,
            width=md.width,
            free_value=md.free_value,
            blocked_value=md.blocked_value,
            food_value=md.food_value,
            snake_tags={k: v for k, v in md.snake_tags.items()},
            snake_values={
                k: simrun_pb2.SnakeValues(head_value=v["head_value"], body_value=v["body_value"])
                for k, v in md.snake_values.items()
            },
            start_positions={k: _coord(v) for k, v in md.start_positions.items()},
            base_map=md.base_map.tobytes(),
            base_map_dtype=str(md.base_map_dtype),
        )
    )
    return proto.SerializeToString()


def _serialize_step(data: LoopStepData) -> bytes:
    proto = simrun_pb2.LoopStepData(
        step=data.step,
        total_time=data.total_time,
        alive_states=dict(data.alive_states),
        snake_times=dict(data.snake_times),
        decisions={k: _coord(v) for k, v in data.decisions.items()},
        tail_directions={k: _coord(v) for k, v in data.tail_directions.items()},
        snake_grew=dict(data.snake_grew),
        lengths=dict(data.lengths),
        new_food=[_coord(c) for c in data.new_food],
        removed_food=[_coord(c) for c in data.removed_food],
    )
    return proto.SerializeToString()


def _serialize_stop(data: LoopStopData) -> bytes:
    proto = simrun_pb2.LoopStopData(final_step=data.final_step)
    return proto.SerializeToString()


class SocketObserver(ILoopObserver):
    """
    Connects to a TCP listener and streams loop events as
    [1 byte: msg_type][4 bytes: big-endian length][N bytes: protobuf].
    """

    def __init__(self, host: str, port: int, connect_timeout_ms: int = 5.0):
        super().__init__()
        self._host = host
        self._port = port
        self._sock = socket.create_connection((host, port), timeout=connect_timeout_ms)
        self._sock.settimeout(None)
        self._queue: Queue = Queue()
        self._stop_event: Event = Event()
        self._failed: Event = Event()
        self._thread = Thread(target=self._send_worker, daemon=True)
        self._thread.start()

    def _send_one(self, msg_type: int, payload: bytes):
        header = struct.pack(">BI", msg_type, len(payload))
        self._sock.sendall(header + payload)

    def _send_worker(self):
        try:
            while not self._stop_event.is_set() or not self._queue.empty():
                try:
                    msg_type, payload = self._queue.get(timeout=0.1)
                except Empty:
                    continue
                try:
                    self._send_one(msg_type, payload)
                except OSError as e:
                    log.error("SocketObserver send failed: %s", e)
                    self._failed.set()
                    return
        finally:
            try:
                self._sock.close()
            except OSError:
                pass

    def has_failed(self) -> bool:
        return self._failed.is_set()

    def notify_start(self, data: LoopStartData):
        if self._failed.is_set():
            return
        self._queue.put((MSG_START, _serialize_start(data)))

    def notify_step(self, data: LoopStepData):
        if self._failed.is_set():
            return
        self._queue.put((MSG_STEP, _serialize_step(data)))

    def notify_stop(self, data: LoopStopData):
        if self._failed.is_set():
            return
        self._queue.put((MSG_STOP, _serialize_stop(data)))

    def reset(self):
        pass  # not meaningful for a one-shot run

    def close(self):
        self._stop_event.set()
        try:
            self._thread.join(timeout=5)
        except Exception:
            pass