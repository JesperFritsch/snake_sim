import tensorflow as tf
import numpy as np
import time
import random
from tensorflow import keras
from collections import deque
from snakes.snake import Snake

class DqnSnake(Snake):
    def __init__(self, id: str, start_length: int):
        super().__init__(id, start_length)

    def update(self):
        norm_map = self.env.get_normalized_map(self.id)


    def reset(self):
        return super().reset()