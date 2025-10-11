
import numpy as np
import struct
import logging
import pickle

from functools import wraps
from pathlib import Path

from snake_sim.storing.interfaces.run_file_handler_interface import IRunFileHandler
from snake_sim.protobuf import simrun_pb2 as run_proto
from snake_sim.protobuf.simrun_pb2 import (
    RunRecord,
    LoopStartData as ProtoLoopStartData,
    LoopStepData as ProtoLoopStepData,
    LoopStopData as ProtoLoopStopData,
    Coord as ProtoCoord,
    EnvMetaData as ProtoEnvMetaData,
    SnakeValues as ProtoSnakeValues
)

from snake_sim.environment.types import (
    LoopStartData,
    LoopStepData,
    LoopStopData,
    EnvMetaData,
    Coord
)

log = logging.getLogger(Path(__file__).stem)

class ProtoRunFileHandler(IRunFileHandler):

    file_format = ".run_proto"

    """ Handles reading and writing of run files. """
    def __init__(self, filepath: str | Path, new: bool = False):
        # if tmp_buffer_size <= 0, no buffering is done
        self._filepath = Path(filepath)
        self._create_new = new
        self._read_only = not new
        self._run_record: run_proto.RunRecord = self._get_run_record()

    def _check_read_only(func):
        @wraps(func)
        def _check_read_only_wrapper(self: "ProtoRunFileHandler", *args, **kwargs):
            if self._read_only:
                raise RuntimeError("FileHandler opened in read-only mode.")
            return func(self, *args, **kwargs)
        return _check_read_only_wrapper

    @_check_read_only
    def write_start(self, start_data: LoopStartData):
        self._run_record.start.CopyFrom(self._start_to_proto(start_data))

    @_check_read_only
    def write_step(self, step_data: LoopStepData):
        step_proto = self._step_to_proto(step_data)
        self._run_record.steps.append(step_proto)

    @_check_read_only
    def write_stop(self, stop_data: LoopStopData):
        stop_proto = self._stop_to_proto(stop_data)
        self._run_record.stop.CopyFrom(stop_proto)

    def get_start_data(self):
        if self._run_record is None or not self._run_record.HasField('start'):
            raise RuntimeError("No start data in run file.")
        return self._proto_to_start(self._run_record.start)

    def iter_steps(self):
        if self._run_record is None:
            return
        for step_proto in self._run_record.steps:
            yield self._proto_to_step(step_proto)

    def get_stop_data(self):
        if self._run_record is None or not self._run_record.HasField('stop'):
            raise RuntimeError("No stop data in run file.")
        return self._proto_to_stop(self._run_record.stop)

    def _get_run_record(self) -> run_proto.RunRecord:
        if self._create_new:
            return RunRecord()
        else:
            return self._read_runfile()

    def _read_runfile(self) -> run_proto.RunRecord:
       with open(self._filepath, "rb") as f:
           run_record = RunRecord()
           run_record.ParseFromString(f.read())
           return run_record

    def _write_runfile(self, run_record: run_proto.RunRecord):
        self._filepath.parent.mkdir(parents=True, exist_ok=True)
        self._filepath = self._filepath.with_suffix(self.file_format)
        logging.info(f"Writing run file to '{self._filepath}'")
        with open(self._filepath, "wb") as f:
            f.write(run_record.SerializeToString())

    def _coord_to_proto(self, coord: Coord) -> run_proto.Coord:
        p = run_proto.Coord()
        p.x = int(coord[0])
        p.y = int(coord[1])
        return p

    def _envinit_to_proto(self, env: EnvMetaData) -> run_proto.EnvMetaData:
        p = run_proto.EnvMetaData()
        p.height = int(env.height)
        p.width = int(env.width)
        p.free_value = int(env.free_value)
        p.blocked_value = int(env.blocked_value)
        p.food_value = int(env.food_value)
        for sid, sval in env.snake_values.items():
            sv = run_proto.SnakeValues()
            sv.head_value = sval['head_value']
            sv.body_value = sval['body_value']
            p.snake_values[int(sid)].CopyFrom(sv)
        for sid, coord in env.start_positions.items():
            c = p.start_positions[int(sid)]
            c.x = int(coord[0])
            c.y = int(coord[1])
        self._write_base_map(p, env.base_map)
        return p

    def _start_to_proto(self, start_data: LoopStartData) -> run_proto.LoopStartData:
        start_proto = run_proto.LoopStartData()
        start_proto.env_meta_data.CopyFrom(self._envinit_to_proto(start_data.env_meta_data))
        return start_proto

    def _step_to_proto(self, step_data: LoopStepData) -> run_proto.LoopStepData:
        step_proto = run_proto.LoopStepData()
        step_proto.step = int(step_data.step)
        step_proto.total_time = float(step_data.total_time)

        # snake_times: Dict[int, float]
        for sid, t in (step_data.snake_times or {}).items():
            step_proto.snake_times[int(sid)] = float(t)

        # decisions / tail_directions / snake_grew / lengths (maps)
        for sid, coord in (step_data.decisions or {}).items():
            c = step_proto.decisions[int(sid)]
            c.x = int(coord[0]); c.y = int(coord[1])

        for sid, coord in (step_data.tail_directions or {}).items():
            c = step_proto.tail_directions[int(sid)]
            c.x = int(coord[0]); c.y = int(coord[1])

        for sid, grew in (step_data.snake_grew or {}).items():
            step_proto.snake_grew[int(sid)] = bool(grew)

        for sid, length in (step_data.lengths or {}).items():
            step_proto.lengths[int(sid)] = int(length)

        # repeated Coord lists
        for coord in (step_data.new_food or []):
            c = step_proto.new_food.add()
            c.x = int(coord[0]); c.y = int(coord[1])
        for coord in (step_data.removed_food or []):
            c = step_proto.removed_food.add()
            c.x = int(coord[0]); c.y = int(coord[1])

        return step_proto

    def _stop_to_proto(self, stop_data: LoopStopData) -> run_proto.LoopStopData:
        stop_proto = run_proto.LoopStopData()
        stop_proto.final_step = int(stop_data.final_step)
        return stop_proto

    def _proto_to_coord(self, p: ProtoCoord) -> Coord:
        return Coord(p.x, p.y)

    def _proto_to_envinit(self, p: ProtoEnvMetaData) -> EnvMetaData:
        snake_values = {sid: {'head_value': sv.head_value, 'body_value': sv.body_value} for sid, sv in p.snake_values.items()}
        start_positions = {sid: self._proto_to_coord(c) for sid, c in p.start_positions.items()}
        base_map = self._read_base_map(p)
        return EnvMetaData(
            height=p.height,
            width=p.width,
            free_value=p.free_value,
            blocked_value=p.blocked_value,
            food_value=p.food_value,
            snake_values=snake_values,
            start_positions=start_positions,
            base_map=base_map,
            base_map_dtype=base_map.dtype
        )

    def _proto_to_start(self, p: ProtoLoopStartData) -> LoopStartData:
        return LoopStartData(
            env_meta_data=self._proto_to_envinit(p.env_meta_data)
        )

    def _proto_to_step(self, p: ProtoLoopStepData) -> LoopStepData:
        snake_times = {sid: t for sid, t in p.snake_times.items()} if p.snake_times else {}
        decisions = {sid: self._proto_to_coord(c) for sid, c in p.decisions.items()} if p.decisions else {}
        tail_directions = {sid: self._proto_to_coord(c) for sid, c in p.tail_directions.items()} if p.tail_directions else {}
        snake_grew = {sid: grew for sid, grew in p.snake_grew.items()} if p.snake_grew else {}
        lengths = {sid: length for sid, length in p.lengths.items()} if p.lengths else {}
        new_food = [self._proto_to_coord(c) for c in p.new_food] if p.new_food else []
        removed_food = [self._proto_to_coord(c) for c in p.removed_food] if p.removed_food else []
        return LoopStepData(
            step=p.step,
            total_time=p.total_time,
            snake_times=snake_times,
            decisions=decisions,
            tail_directions=tail_directions,
            snake_grew=snake_grew,
            lengths=lengths,
            new_food=new_food,
            removed_food=removed_food
        )

    def _write_base_map(self, proto_env, env_data: EnvMetaData):
        arr = env_data.base_map
        target_dtype = env_data.base_map_dtype
        arr2 = arr.astype(target_dtype, copy=False)
        arr2 = np.ascontiguousarray(arr2)
        proto_env.base_map = arr2.tobytes(order='C')
        proto_env.base_map_dtype = arr2.dtype.str

    def _read_base_map(self, proto_env: ProtoEnvMetaData) -> np.ndarray:
        dtype = np.dtype(proto_env.base_map_dtype) if proto_env.base_map_dtype else np.uint8
        arr = np.frombuffer(proto_env.base_map, dtype=dtype)
        arr = arr.reshape((proto_env.height, proto_env.width), order='C')
        if arr.dtype.byteorder not in ('=', '|'):
            arr = arr.byteswap().newbyteorder()
        return arr

    def _proto_to_stop(self, p: ProtoLoopStopData) -> LoopStopData:
        return LoopStopData(
            final_step=p.final_step
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._create_new and self._run_record is not None:
            self._write_runfile(self._run_record)
        return