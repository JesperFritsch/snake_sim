"""
RL data bus singleton to share data between RL components. 
E.g., to share episode states between the environment and the trainer.
"""

from utils import SingletonMeta
from snake_sim.rl.types import RLTransitionData

class RLMetaDataBus(metaclass=SingletonMeta):
    def __init__(self):
        self.transitions = []

    def add_transition(self, transition: RLTransitionData):
        self.transitions.append(transition)

    def get_transitions(self):
        return self.transitions
