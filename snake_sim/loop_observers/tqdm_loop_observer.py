import sys
from snake_sim.environment.interfaces.loop_observer_interface import ILoopObserver
from snake_sim.environment.types import LoopStepData
from tqdm import tqdm


class TqdmLoopObserver(ILoopObserver):

    def __init__(self):
        self._pbar = None

    def notify_start(self):
        self._pbar = tqdm(desc='Simulation steps', unit='step')

    def notify_step(self, step_data: LoopStepData):
        self._pbar.update(1)
        longest_snake = max(step_data.lengths.items(), key=lambda item: item[1]) 
        nr_alive = sum(1 for v in step_data.decisions)
        self._pbar.set_postfix(nr_alive=f"nr alive: {nr_alive}", longest=f"{longest_snake[0]} ({longest_snake[1]})")

    def notify_end(self):
        if self._pbar is not None:
            self._pbar.close()
            print()
            sys.stdout.flush()
            self._pbar = None