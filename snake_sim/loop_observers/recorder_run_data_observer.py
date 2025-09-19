
from typing import Optional
from pathlib import Path

from snake_sim.environment.interfaces.run_data_observer_interface import IRunDataObserver
from snake_sim.run_data.run_data import StepData, RunData

class RecorderRunDataObserver(IRunDataObserver):
    def __init__(self, recording_dir: Optional[str] = None, recording_file: Optional[str]=None):
        self.recording_file = recording_file
        self.recording_dir = recording_dir

    def notify_start(self, metadata: dict):
        pass

    def notify_step(self, step_data: StepData):
        pass

    def notify_end(self, run_data: RunData):
        run_dir = Path(self.recording_dir).joinpath(f'grid_{run_data.width}x{run_data.height}')
        run_data.write_to_file(run_dir, filename=self.recording_file)
