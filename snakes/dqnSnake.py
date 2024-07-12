import tensorflow as tf
import numpy as np
import time
import random
from pathlib import Path
from utils import coord_op
from tensorflow import keras
from tensorflow.keras import models, layers
from collections import deque
from snakes.snake import Snake
from snake_env import DIR_MAPPING
# from multiprocessing import Process, Pipe


def agent2():
    model = models.Sequential()
    # Add a convolutional layer
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(20, 20, 1)))  # 20x20 grid, 1 channel (grayscale)
    # Add another convolutional layer
    # Flatten the output and add a fully connected layer
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    # Output layer (assuming a classification task, e.g., predicting next move)
    model.add(layers.Dense(len(DIR_MAPPING), activation='softmax'))  # num_classes: number of possible actions/moves
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])


class DqnSnake(Snake):
    def __init__(self, id: str, start_length: int, training: bool = False):
        super().__init__(id, start_length)
        self.state_stack = deque(maxlen=4)
        self.training = training
        self.model = agent2((self.env.height, self.env.width, 4), 4)
        self.target_model = agent2((self.env.height, self.env.width, 4), 4)
        self.replay_memory = deque(maxlen=50_000)
        self.weights_file = Path(__file__).parent / 'ml' / 'weights' / 'dqn_weights.h5'
        self.prev_map_state = None
        self.last_action = None
        self.steps = 0
        self.total_reward = 0
        self.steps_to_update_target = 100
        self.steps_to_train = 4
        self.replay_batch_size = 32
        self.max_epsilon = 1.0
        self.min_epsilon = 0.01
        self.epsilon = self.max_epsilon
        self.epsilon_decay = 0.01
        self.gamma = 0.95 # discount factor
        self.alpha = 0.5 # learning rate

    def update(self, reward_last_step: float):
        self.steps += 1
        self.update_map(self.env.map)
        new_map_state = self.env.get_expanded_normal_map(self.id)
        # print(norm_map)
        if self.training:
            if self.steps % self.steps_to_train == 0:
                step = (self.prev_map_state, self.last_action, reward_last_step, new_map_state)
                self.replay_memory.append(step)
                self.total_reward += reward_last_step
                self.train()

            if self.steps % self.steps_to_update_target == 0:
                self.update_target_model()
            self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.epsilon_decay * self.steps)
        if np.random.rand() < self.epsilon:
            acutal_action = random.choice(list(DIR_MAPPING.keys()))
            action_index = DIR_MAPPING.keys().index(acutal_action)
        else:
            predicted = self.model(new_map_state).flatten()
            action_index = np.argmax(predicted)
        self.last_action = action_index
        return DIR_MAPPING.keys()[action_index]

    def train(self):
        if len(self.replay_memory) < self.replay_batch_size:
            return

        batch = random.sample(self.replay_memory, self.replay_batch_size)
        initial_map_states = np.array([state[0] for state in batch])
        next_states = np.array([state[3] for state in batch])
        current_qs_list = self.model.predict(initial_map_states)
        next_qs_list = self.target_model.predict(next_states)
        X = []
        Y = []
        for index, (state, action, reward, next_state) in enumerate(batch):
            max_future_q = reward + self.gamma * np.max(next_qs_list[index])
            current_qs = current_qs_list[index]
            # Update the Q value for the given state
            current_qs[action] = (1 - self.alpha) * current_qs[action] + self.alpha * max_future_q
            X.append(state)
            Y.append(current_qs)

        self.model.fit(np.array(X), np.array(Y), epochs=1, verbose=0)

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def save_weights(self):
        self.model.save_weights(self.weights_file)

    def load_weights(self, model):
        model.load_weights(self.weights_file)

    def kill(self):
        self.save_weights()
        return super().kill()

    def reset(self):
        self.steps = 0
        return super().reset()