from pathlib import Path

from snake_sim.storing.interfaces.run_file_handler_interface import IRunFileHandler
from snake_sim.storing.pickle_run_file_handler import PickleRunFileHandler
from snake_sim.storing.proto_run_file_handler import ProtoRunFileHandler
from snake_sim.storing.json_run_file_handler import JsonRunFileHandler


def create_run_file_handler(
        filepath: str | Path=None,
        *args, **kwargs) -> IRunFileHandler:
    file_format = '.' + Path(filepath).suffix.lstrip(".")
    for handler in (
        PickleRunFileHandler, 
        ProtoRunFileHandler, 
        JsonRunFileHandler
    ):
        if handler._get_file_format() == file_format:
            print(f"Found handler {type(handler)} for format {file_format}")
            return handler(filepath=filepath, *args, **kwargs)
    else:
        raise ValueError(f"Unknown file format: {file_format}")


