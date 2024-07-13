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
# from multiprocessing import Process, Pipe

ACTIONS = ((0, -1), (1,  0), (0,  1), (-1, 0))


def agent2():
    model = models.Sequential()
    model.add(layers.Input(shape=(64, 64, 1)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(len(ACTIONS)))
    model.compile(optimizer='adam', loss='mse')
    return model


class DqnSnake(Snake):
    def __init__(self, id: str, start_length: int, training: bool = False):
        super().__init__(id, start_length)
        self.state_stack = deque(maxlen=4)
        self.training = training
        self.model = agent2()
        self.target_model = agent2()
        self.replay_memory = deque(maxlen=50_000)
        self.weights_file = Path(__file__).parent.parent / 'ml' / 'weight_files' / 'dqn.weights.h5'
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
        self.gamma = 0.95
        self.alpha = 0.5
        self.rewards = []
        if self.weights_file.exists():
            self.load_weights(self.model)
            self.load_weights(self.target_model)
        self.model.summary()

    def update(self):
        self.steps += 1
        self.update_map(self.env.map)
        new_map_state = self.env.get_expanded_normal_map(self.id)
        if self.prev_map_state is None:
            self.prev_map_state = new_map_state

        if self.training:
            last_reward = self.calculate_reward()
            self.rewards.append(last_reward)
            step = (self.prev_map_state, self.last_action, last_reward, new_map_state)
            self.replay_memory.append(step)
            self.total_reward += last_reward
            if self.steps % self.steps_to_train == 0:
                self.train()

            if self.steps % self.steps_to_update_target == 0:
                self.update_target_model()
            self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.epsilon_decay * self.steps)


        if np.random.rand() < self.epsilon and self.training:
            acutal_action = random.choice(ACTIONS)
            action_index = ACTIONS.index(acutal_action)
        else:
            # print('new_map_state: ', new_map_state.shape)
            new_map_state.reshape(1, 64, 64, 1)
            predicted = self.model.predict(new_map_state.reshape(1, 64, 64, 1))
            action_index = np.argmax(predicted)
        self.last_action = action_index
        self.prev_map_state = new_map_state
        return ACTIONS[action_index]


    def train(self):
        if len(self.replay_memory) < self.replay_batch_size:
            return
        batch = random.sample(self.replay_memory, self.replay_batch_size)
        self.fit_batch(batch)

    def final_train(self):
        if len(self.replay_memory) < self.replay_batch_size:
            return
        batch = list(self.replay_memory)[self.replay_batch_size:]
        self.fit_batch(batch)

    def fit_batch(self, batch):
        initial_map_states = np.array([state[0] for state in batch])
        next_states = np.array([state[3] for state in batch])

        # Ensure the correct format for the states

        # print('initial_map_states: ', initial_map_states.shape)
        # print('next_states: ', next_states.shape)
        current_qs_list = self.model.predict(initial_map_states)
        next_qs_list = self.target_model.predict(next_states)

        X = []
        Y = []
        for index, (state, action, reward, next_state) in enumerate(batch):
            max_future_q = reward + self.gamma * np.max(next_qs_list[index])
            current_qs = current_qs_list[index]
            current_qs[action] = (1 - self.alpha) * current_qs[action] + self.alpha * max_future_q
            X.append(state)
            Y.append(current_qs)

        X = np.array(X)
        Y = np.array(Y)
        self.model.train_on_batch(X, Y)

    def calculate_reward(self):
        steps_without_food = self.env.time_step - self.env.snakes_info[self.id]['last_food']
        last_food = self.env.snakes_info[self.id]['last_food']
        food_reward = 1 if last_food == self.env.time_step else 0
        eaten_reward = 0.05 * (20 - min(steps_without_food, 20))
        dead_reward = 0.1 if self.env.snakes_info[self.id]['alive'] else -1
        return food_reward + eaten_reward + dead_reward

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def save_weights(self):
        self.model.save_weights(self.weights_file)

    def load_weights(self, model):
        model.load_weights(self.weights_file)

    def kill(self):
        new_map_state = self.env.get_expanded_normal_map(self.id)
        step = (self.prev_map_state, self.last_action, -1, new_map_state)
        self.replay_memory.append(step)
        self.final_train()
        self.save_weights()
        self.rewards.append(self.total_reward)  # Log total reward for the episode
        return super().kill()

    def reset(self):
        # self.steps = 0
        self.rewards = []
        self.total_reward = 0
        return super().reset()
