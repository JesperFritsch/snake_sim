import os
import struct
import logging
import json

from pathlib import Path

from snake_sim.storing.interfaces.run_file_handler_interface import IRunFileHandler

from snake_sim.environment.types import (
    LoopStartData,
    LoopStepData,
    LoopStopData
)

type LoopData = LoopStartData | LoopStepData | LoopStopData


log = logging.getLogger(Path(__file__).stem)

class JsonRunFileHandler(IRunFileHandler):

    file_format = ".json"

    """ Handles reading and writing of run files. """
    def __init__(self, filepath: str | Path, new: bool = False):
        # if tmp_buffer_size <= 0, no buffering is done
        self._filepath = Path(filepath)
        self._create_new = new
        self._file = None
        self._type_indicators: dict[type[LoopData], str] = {
            LoopStartData: "start",
            LoopStepData: "step",
            LoopStopData: "stop",
        }
        self._type_indicators_rev: dict[str, type[LoopData]] = {v: k for k, v in self._type_indicators.items()}

    def write_start(self, start_data: LoopStartData):
        self._write_line(start_data)

    def write_step(self, step_data: LoopStepData):
        self._write_line(step_data)

    def write_stop(self, stop_data: LoopStopData):
        self._write_line(stop_data)
        
    def get_start_data(self) -> LoopStartData:
        self._file.seek(0)
        return self._parse_line(self._file.readline())

    def iter_steps(self):
        self._file.seek(0)
        self._file.readline()
        for line in self._file:
            if not line.strip():
                continue
            loop_data = self._parse_line(line)
            if isinstance(loop_data, LoopStepData):
                yield loop_data
            else:
                break
            
    def get_stop_data(self) -> LoopStopData:
        last_line = self._last_line()
        return self._parse_line(last_line)

    def _last_line(self) -> str:
        try:
            self._file.seek(-2, os.SEEK_END)        
            while self._file.read(1) != b"\n":      
                self._file.seek(-2, os.SEEK_CUR)
        except OSError:                    
            self._file.seek(0)
        return self._file.readline()

    def _parse_line(self, line: str) -> LoopData:
        obj = json.loads(line)
        data_type = obj["type"]
        cls = self._type_indicators_rev[data_type]
        return cls.model_validate(obj["data"])

    def _write_line(self, object: LoopData):
        type_ind = self._type_indicators[type(object)]
        line_obj = {
            "type": type_ind,
            "data": object.model_dump(mode="json")
        }
        self._file.write(json.dumps(line_obj))
        self._file.write("\n")

    def _open_for_append(self):
        self._file = open(self._filepath, "a")

    def _open_for_write(self):
        self._file = open(self._filepath, "w")

    def _open_for_read(self):
        # Binary so _last_line() can do negative seeks; json.loads handles bytes.
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
        log.debug(f"Closing run file at '{self._filepath}'")
        if self._file:
            self._file.close()
            self._file = None