from snake_sim.environment.interfaces.loop_observer_interface import ILoopObserver
from tqdm import tqdm


class TqdmLoopObserver(ILoopObserver):

    def __init__(self):
        self._pbar = None

    def notify_start(self):
        self._pbar = tqdm(desc='Simulation steps', unit='step')

    def notify_step(self, step_data):
        self._pbar.update(1)

    def notify_end(self):
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None