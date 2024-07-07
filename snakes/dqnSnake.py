# import tensorflow as tf
import numpy as np
import time
import random
from utils import coord_op
# from tensorflow import keras
from collections import deque
from snakes.snake import Snake

def agent(state_shape, action_shape):
    """ The agent maps X-states to Y-actions
    e.g. The neural network output is [.1, .7, .1, .3]
    The highest value 0.7 is the Q-Value.
    The index of the highest action (0.7) is action #1.
    """
    learning_rate = 0.001
    init = tf.keras.initializers.HeUniform()
    model = keras.Sequential()
    model.add(keras.layers.Dense(24, input_shape=state_shape, activation='relu', kernel_initializer=init))
    model.add(keras.layers.Dense(12, activation='relu', kernel_initializer=init))
    model.add(keras.layers.Dense(action_shape, activation='linear', kernel_initializer=init))
    model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])
    return model

class DqnSnake(Snake):
    def __init__(self, id: str, start_length: int, training: bool = False):
        super().__init__(id, start_length)
        self.state_stack = deque(maxlen=4)
        self.training = training
        self.model = agent((self.env.height, self.env.width, 4), 4)
        self.target_model = agent((self.env.height, self.env.width, 4), 4)

    def update(self):
        self.update_map(self.env.map)
        norm_map = self.env.get_expanded_normal_map(self.id)
        print(norm_map)
        if self.training:
            self.state_stack.append(norm_map)
            self.train(norm_map)
        return coord_op(self.coord, (1, 0), '+')

    def init_training(self):
        while len(self.state_stack) < self.state_stack.maxlen:
            self.state_stack.append(np.full((self.env.height, self.env.width), fill_value=self.env.FREE_TILE, dtype=np.uint8))

    def reset(self):
        return super().reset()