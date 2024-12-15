
from typing import Optional

from snake_sim.loop_observers.run_data_observer_interface import IRunDataObserver
from snake_sim.run_data.run_data import StepData, RunData

class RecorderRunDataObserver(IRunDataObserver):
    def __init__(self, recording_file: Optional[str]=None, as_proto=False):
        self.recording_file = recording_file
        self.as_proto = as_proto

    def notify_start(self, metadata: dict):
        pass

    def nofify_step(self, step_data: StepData):
        pass

    def notify_end(self, run_data: RunData):
        run_data.write_to_file(self.recording_file, as_proto=self.as_proto)
