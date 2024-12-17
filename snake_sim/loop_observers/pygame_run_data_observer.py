

from multiprocessing.connection import PipeConnection

from snake_sim.loop_observers.run_data_observer_interface import IRunDataObserver
from snake_sim.run_data.run_data import StepData

class PygameRunDataObserver(IRunDataObserver):
    def __init__(self, pipe_conn: PipeConnection):
        if not isinstance(pipe_conn, PipeConnection):
            raise ValueError('pipe_conn must be of type Connection but is {}'.format(type(pipe_conn)))
        self.pipe_conn = pipe_conn
        self.adapter = None

    def notify_start(self, metadata: dict):
        self.pipe_conn.send(metadata)

    def notify_step(self, step_data: StepData):
        self.pipe_conn.send(step_data.to_dict())

    def notify_end(self, run_data):
        self.pipe_conn.send('stopped')
