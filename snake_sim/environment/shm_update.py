import struct
import time
import logging
from pathlib import Path
from typing import Optional
from multiprocessing import shared_memory
import os
import atexit
import signal
import sys

log = logging.getLogger(Path(__file__).stem)

# Shared memory reader helper
SEQ_OFF = 0
PAYLOAD_SIZE_OFF = 8
READER_COUNT_OFF = 16
BASE_HEADER_SZ = 24
ACK_SLOT_SZ = 8

class SharedMemoryReader:
    """Attach to an existing shared memory region created by the environment writer.

    Header layout (agreed):
      0: seq (u64)         -- odd = writer in progress, even = complete
      8: payload_size (u64)
     16: reader_count (u32)
     20: padding (4 bytes)
     24: ack slots (reader_count * u64)
    payload_offset = 24 + reader_count*8
    """

    def __init__(self, shm_name: str, reader_id: int):
        # Attach to existing shared memory without registering the name
        # with multiprocessing.resource_tracker in this process. This
        # keeps reader/server processes from appearing to 'own' the
        # shared memory and emitting resource_tracker warnings at exit.
        try:
            from multiprocessing import resource_tracker
            orig_register = getattr(resource_tracker, 'register', None)
            if orig_register is not None:
                # temporarily replace register with a no-op
                resource_tracker.register = lambda name, rtype: None
            try:
                self._shm = shared_memory.SharedMemory(name=shm_name)
            finally:
                # restore original register implementation
                if orig_register is not None:
                    resource_tracker.register = orig_register
                # Do NOT call resource_tracker.unregister here. We intentionally
                # prevented local registration above; calling unregister when
                # no registration exists can trigger KeyError traces in the
                # resource_tracker process. Rely on the no-op register to keep
                # reader processes unregistered.
        except Exception:
            # If resource_tracker is unavailable for any reason, fall back
            # to a normal attach and tolerate possible tracker registration.
            self._shm = shared_memory.SharedMemory(name=shm_name)
        self._buf = self._shm.buf
        self._reader_id = int(reader_id)
        # read reader_count from header
        reader_count = struct.unpack_from('<I', self._buf, READER_COUNT_OFF)[0]
        if self._reader_id < 0 or self._reader_id >= reader_count:
            raise ValueError(f"reader_id {self._reader_id} out of range (0..{reader_count-1})")
        self._reader_count = reader_count
        self._payload_offset = BASE_HEADER_SZ + self._reader_count * ACK_SLOT_SZ

    @property
    def reader_id(self) -> int:
        return self._reader_id

    def close(self):
        log.debug("SharedMemoryReader.close name=%s pid=%s", self._shm.name, os.getpid())
        try:
            self._shm.close()
        except Exception:
            # be tolerant on close
            logging.exception("SharedMemoryReader.close failed")

    def _read_seq(self) -> int:
        return struct.unpack_from('<Q', self._buf, SEQ_OFF)[0]

    def _read_payload_size(self) -> int:
        return struct.unpack_from('<Q', self._buf, PAYLOAD_SIZE_OFF)[0]

    def _write_ack(self, seq: int) -> None:
        slot_off = BASE_HEADER_SZ + self._reader_id * ACK_SLOT_SZ
        struct.pack_into('<Q', self._buf, slot_off, seq)

    def read_frame(self, max_retries: int = 5, retry_sleep: float = 0.001) -> Optional[bytes]:
        """Attempt to read a consistent payload from shared memory using seq check.
        Returns payload bytes on success or None on timeout.
        """
        for _ in range(max_retries):
            seq1 = self._read_seq()
            if seq1 & 1:
                time.sleep(retry_sleep)
                continue
            payload_size = self._read_payload_size()
            # copy payload
            start = self._payload_offset
            end = start + payload_size
            payload = bytes(self._buf[start:end])
            seq2 = self._read_seq()
            if seq1 == seq2 and (seq2 & 1) == 0:
                # publish ack
                try:
                    self._write_ack(seq2)
                except Exception:
                    logging.exception("Failed to write ack slot")
                return payload
            time.sleep(retry_sleep)
        return None

# Shared memory writer helper (placed in this file to keep reader/writer layout together)
class SharedMemoryWriter:
    """Create and write frames into a shared memory region that matches the reader layout above.

    Header layout (agreed):
      0: seq (u64)         -- odd = writer in progress, even = complete
      8: payload_size (u64)
     16: reader_count (u32)
     20: padding (4 bytes)
     24: ack slots (reader_count * u64)
    payload_offset = 24 + reader_count*8
    """

    def __init__(self, shm: shared_memory.SharedMemory, reader_count: int, payload_capacity: int):
        self._shm = shm
        self._buf = shm.buf
        self._reader_count = int(reader_count)
        self._payload_capacity = int(payload_capacity)
        self._payload_offset = BASE_HEADER_SZ + self._reader_count * ACK_SLOT_SZ
        self._unlinked = True

    @classmethod
    def create(cls, name: str, reader_count: int, payload_capacity: int, overwrite: bool = False):
        total_size = BASE_HEADER_SZ + reader_count * ACK_SLOT_SZ + payload_capacity
        try:
            shm = shared_memory.SharedMemory(name=name, create=True, size=total_size)
        except FileExistsError:
            if overwrite:
                try:
                    existing = shared_memory.SharedMemory(name=name)
                    existing.close()
                    existing.unlink()
                except Exception:
                    pass
                shm = shared_memory.SharedMemory(name=name, create=True, size=total_size)
            else:
                raise

        writer = cls(shm, reader_count, payload_capacity)
        # mark ownership: the creating process is responsible for unlinking
        writer._unlinked = False

        # initialize header
        writer._write_seq(0)
        writer._write_payload_size(0)
        struct.pack_into('<I', writer._buf, READER_COUNT_OFF, reader_count)
        # zero ack slots
        for i in range(reader_count):
            slot_off = BASE_HEADER_SZ + i * ACK_SLOT_SZ
            struct.pack_into('<Q', writer._buf, slot_off, 0)
        return writer

    def close(self):
        try:
            log.debug("SharedMemoryWriter.close name=%s pid=%s", self._shm.name, os.getpid())
            self._shm.close()
        except Exception:
            logging.exception("SharedMemoryWriter.close failed")

    def unlink(self):
        # idempotent unlink - best-effort
        if getattr(self, '_unlinked', False):
            log.debug("SharedMemoryWriter.unlink already unlinked name=%s pid=%s", self._shm.name, os.getpid())
            return
        name = getattr(self._shm, 'name', None)
        try:
            self._shm.unlink()
            self._unlinked = True
            log.debug("SharedMemoryWriter.unlink succeeded name=%s pid=%s", name, os.getpid())
        except FileNotFoundError:
            # already unlinked by someone else; treat as unlinked
            self._unlinked = True
            log.debug("SharedMemoryWriter.unlink FileNotFoundError name=%s pid=%s", name, os.getpid())
        except Exception:
            # log at debug level and ignore - best-effort cleanup
            logging.exception('Failed to unlink shared memory')
            self._unlinked = True

    def cleanup(self):
        log.debug("SharedMemoryWriter.cleanup name=%s pid=%s", self._shm.name, os.getpid())
        self.close()
        self.unlink()

    def _read_seq(self) -> int:
        return struct.unpack_from('<Q', self._buf, SEQ_OFF)[0]

    def _write_seq(self, val: int) -> None:
        struct.pack_into('<Q', self._buf, SEQ_OFF, int(val))

    def _write_payload_size(self, val: int) -> None:
        struct.pack_into('<Q', self._buf, PAYLOAD_SIZE_OFF, int(val))

    def write_frame(self, payload: bytes) -> int:
        """Write a payload to shared memory and return the published seq (even).
        """
        if len(payload) > self._payload_capacity:
            raise ValueError(f"payload size {len(payload)} exceeds capacity {self._payload_capacity}")

        seq = self._read_seq()
        # mark writing: odd value
        writing_seq = seq + 1
        self._write_seq(writing_seq)
        # write payload size
        self._write_payload_size(len(payload))
        # write payload bytes
        start = self._payload_offset
        end = start + len(payload)
        self._buf[start:end] = payload
        # mark done: even value
        published_seq = seq + 2
        self._write_seq(published_seq)
        return published_seq

    def _wait_for_acks(self, seq: int, timeout: float) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            all_ok = True
            for i in range(self._reader_count):
                slot_off = BASE_HEADER_SZ + i * ACK_SLOT_SZ
                v = struct.unpack_from('<Q', self._buf, slot_off)[0]
                if v != seq:
                    all_ok = False
                    break
            if all_ok:
                return True
            time.sleep(0.001)
        return False


def create_shm_writer(name: str, reader_count: int, payload_capacity: int, overwrite: bool = False) -> SharedMemoryWriter:
    return SharedMemoryWriter.create(name, reader_count, payload_capacity, overwrite=overwrite)


