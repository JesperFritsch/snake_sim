
import struct
import logging
import pickle

from pathlib import Path

from snake_sim.storing.interfaces.run_file_handler_interface import IRunFileHandler

from snake_sim.environment.types import (
    LoopStartData,
    LoopStepData,
    LoopStopData
)

log = logging.getLogger(Path(__file__).stem)

class RunFileHandler(IRunFileHandler):

    file_format = ".run"

    """ Handles reading and writing of run files. """
    def __init__(self, filepath: str | Path, new: bool = False, tmp_buffer_size: int = 1 << 20):
        # if tmp_buffer_size <= 0, no buffering is done
        self._filepath = Path(filepath)
        self._create_new = new
        self._file = None
        self._hdr_struct = struct.Struct("!BI")
        self._type_indicators = {
            LoopStartData: 1,
            LoopStepData: 2,
            LoopStopData: 3
        }
        self._type_indicators_rev = {v: k for k, v in self._type_indicators.items()}
        self._tmp_buffer_size = tmp_buffer_size
        self._tmp_buffer = bytearray()

    def append_object(self, w_object: LoopStartData | LoopStepData | LoopStopData):
        type_indicator = self._type_indicators[type(w_object)]
        obj_bytes = pickle.dumps(w_object, protocol=pickle.HIGHEST_PROTOCOL)
        self._tmp_buffer.extend(self._hdr_struct.pack(type_indicator, len(obj_bytes)))
        self._tmp_buffer.extend(obj_bytes)
        self._write_buffer()

    def _write_buffer(self, force=False):
        if self._tmp_buffer_size > 0 and (len(self._tmp_buffer) > self._tmp_buffer_size or force):
            self._file.write(self._tmp_buffer)
            self._tmp_buffer.clear()

    def write_start(self, start_data: LoopStartData):
        self.append_object(start_data)

    def write_step(self, step_data: LoopStepData):
        self.append_object(step_data)

    def write_stop(self, stop_data: LoopStopData):
        self.append_object(stop_data)

    def get_start_data(self) -> LoopStartData:
        hdr_data = self._file.read(self._hdr_struct.size)
        type_indicator, obj_len = self._hdr_struct.unpack(hdr_data)
        if type_indicator != self._type_indicators[LoopStartData]:
            raise RuntimeError("File does not start with LoopStartData.")
        obj_data = self._file.read(obj_len)
        if len(obj_data) != obj_len:
            raise RuntimeError("File ended unexpectedly while reading LoopStartData.")
        return pickle.loads(obj_data)

    def iter_steps(self):
        while True:
            hdr_data = self._file.read(self._hdr_struct.size)
            if not hdr_data:
                break
            type_indicator, obj_len = self._hdr_struct.unpack(hdr_data)
            obj_data = self._file.read(obj_len)
            if len(obj_data) != obj_len:
                raise RuntimeError("File ended unexpectedly while reading object data.")
            obj_type = self._type_indicators_rev.get(type_indicator)
            if obj_type is None:
                raise RuntimeError(f"Unknown type indicator {type_indicator} in file.")
            if obj_type == LoopStopData:
                # Stop iteration when we reach LoopStopData
                break
            yield pickle.loads(obj_data)

    def get_stop_data(self) -> LoopStopData:
        #first find the stop data
        self._file.seek(0)
        while True:
            hdr_data = self._file.read(self._hdr_struct.size)
            type_indicator, obj_len = self._hdr_struct.unpack(hdr_data)
            if type_indicator == self._type_indicators[LoopStopData]:
                obj_data = self._file.read(obj_len)
                if len(obj_data) != obj_len:
                    raise RuntimeError("File ended unexpectedly while reading LoopStopData.")
                return pickle.loads(obj_data)
            else:
                self._file.seek(obj_len, 1)

    def _open_for_append(self):
        self._file = open(self._filepath, "b+a")

    def _open_for_write(self):
        self._file = open(self._filepath, "wb")

    def _open_for_read(self):
        self._file = open(self._filepath, "rb")

    def __enter__(self):
        log.debug(f"Opening run file at '{self._filepath}' (new={self._create_new})")
        if self._create_new:
            self._filepath.parent.mkdir(parents=True, exist_ok=True)
            self._filepath = self._filepath.with_suffix(self.file_format)
            self._open_for_write()
        else:
            self._open_for_read()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._create_new:
            self._write_buffer(force=True)
        log.debug(f"Closing run file at '{self._filepath}'")
        if self._file:
            self._file.close()
            self._file = None