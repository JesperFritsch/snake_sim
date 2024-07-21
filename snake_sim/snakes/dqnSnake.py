import tensorflow as tf
import numpy as np
import time
import random
from pathlib import Path
from snake_sim.utils import coord_op
from tensorflow import keras
from tensorflow.keras import models, layers
from collections import deque
from snake_sim.snakes.autoSnakeBase import AutoSnakeBase
from snake_sim.snake_env import SnakeEnv
# from multiprocessing import Process, Pipe

ACTIONS = ((0, -1), (1,  0), (0,  1), (-1, 0))
print_every = 100


def agent2(width, height):
    model = models.Sequential()
    model.add(layers.Input(shape=(height, width, 1)))
    model.add(layers.Conv2D(32, (3, 3), padding='same'))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(len(ACTIONS)))
    model.compile(optimizer='adam', loss='mse')
    return model


class DqnSnake(AutoSnakeBase):
    def __init__(self, id: str, start_length: int, height: int, width: int, training: bool = False):
        super().__init__(id, start_length)
        self.in_height = height
        self.in_width = width
        self.state_stack = deque(maxlen=4)
        self.training = training
        self.model = agent2(self.in_width, self.in_height)
        self.target_model = agent2(self.in_width, self.in_height)
        self.replay_memory = deque(maxlen=20_000)
        self.weights_file = Path(__file__).parent.parent.parent / 'ml' / 'weight_files' / f'dqn{self.in_height}x{self.in_width}_colab.weights.h5'
        self.prev_map_state = None
        self.last_action = 0
        self.steps = 0
        self.training_steps = 0
        self.total_reward = 0
        self.total_rewards = []
        self.lengths = []
        self.nr_of_steps = []
        self.steps_to_update_target = 100
        self.steps_to_train = 4
        self.replay_batch_size = 32
        self.max_epsilon = 1.0
        self.min_epsilon = 0.01
        self.epsilon = self.max_epsilon
        self.epsilon_decay = 0.0005
        self.gamma = 0.95
        self.alpha = 0.5
        self.dead_reward = -1
        if self.weights_file.exists():
            self.load_weights(self.model)
            self.load_weights(self.target_model)
        self.model.summary()

    def update(self):
        self.steps += 1
        self.training_steps += 1
        self.update_map(self.env.map)
        new_map_state = self.env.get_expanded_normal_map(self.id)
        if self.prev_map_state is None:
            self.prev_map_state = new_map_state

        if self.training:
            last_reward = self.calculate_reward()
            step = (self.prev_map_state, self.last_action, last_reward, new_map_state, False)
            self.replay_memory.append(step)

            # For every step, add a bad step to the replay memory to help the snake learn
            opposite_action = (self.last_action + 2) % 4
            bad_step = (self.prev_map_state, opposite_action, -0.3, self.prev_map_state, True)
            self.replay_memory.append(bad_step)

            self.total_reward += last_reward
            if self.training_steps % self.steps_to_train == 0:
                self.train()

            if self.training_steps % self.steps_to_update_target == 0:
                self.update_target_model()
            self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.epsilon_decay * self.training_steps)
            if self.training_steps % print_every == 0:
                avg_reward = sum(self.total_rewards[-print_every:]) / print_every
                avg_steps_taken = sum(self.nr_of_steps[-print_every:]) / print_every
                avg_length = sum(self.lengths[-print_every:]) / print_every
                max_reward = max(self.total_rewards[-print_every:])
                max_steps_taken = max(self.nr_of_steps[-print_every:])
                max_length = max(self.lengths[-print_every:])
                print(f"{f' Episode: {len(self.total_rewards)} ':#^50}")
                print(f'Avg reward: {avg_reward}, Avg steps taken: {avg_steps_taken}, Avg length: {avg_length}, Max reward: {max_reward}, Max steps taken: {max_steps_taken}, Max length: {max_length}')

        if np.random.rand() < self.epsilon and self.training:
            acutal_action = random.choice(ACTIONS)
            action_index = ACTIONS.index(acutal_action)
        else:
            # print('new_map_state: ', new_map_state.shape)
            predicted = self.model(new_map_state.reshape(1, self.in_height, self.in_width, 1))
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
        batch = list(self.replay_memory)[-self.replay_batch_size:]
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
        for index, (state, action, reward, next_state, dead) in enumerate(batch):
            if dead:
                max_future_q = reward
            else:
                max_future_q = reward + self.gamma * np.max(next_qs_list[index])

            current_qs = current_qs_list[index]
            current_qs[action] = (1 - self.alpha) * current_qs[action] + self.alpha * max_future_q
            X.append(state)
            Y.append(current_qs)

        X = np.array(X)
        Y = np.array(Y)
        self.model.train_on_batch(X, Y)

    def calculate_reward(self):
        last_food = self.env.snakes_info[self.id]['last_food']
        # steps_without_food = self.env.time_step - last_food
        food_reward = 1 if last_food == self.env.time_step - 1 else 0
        current_head = self.coord
        distance_to_food = self.distance_to_food(current_head)
        last_head = self.body_coords[1]
        last_distance_to_food = self.distance_to_food(last_head)
        # self.print_map(self.map)
        # print('distance_to_food: ', distance_to_food)
        # print('last_distance_to_food: ', last_distance_to_food)

        food_distance_reward = 0.6
        if (distance_to_food is not None and last_distance_to_food is not None) and distance_to_food >= last_distance_to_food:
            food_distance_reward = -food_distance_reward
        return food_reward + food_distance_reward

    def distance_to_food(self, from_coord):
        checked = np.full((self.env.height, self.env.width), fill_value=False, dtype=bool)
        current_coords = [from_coord]
        s_map = self.map.copy()
        distance = 0
        while current_coords:
            next_coords = []
            for coord in current_coords:
                if s_map[coord[1], coord[0]] == SnakeEnv.FOOD_TILE:
                    return distance
                valid_tiles = self.valid_tiles(s_map, coord)
                for valid_coord in valid_tiles:
                    t_x, t_y = valid_coord
                    if not checked[t_y, t_x]:
                        next_coords.append(valid_coord)
                        checked[t_y, t_x] = True
            current_coords = next_coords
            distance += 1
        return None

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def save_weights(self):
        self.model.save_weights(self.weights_file)

    def load_weights(self, model):
        model.load_weights(self.weights_file)

    def kill(self):
        new_map_state = self.env.get_expanded_normal_map(self.id)
        step = (self.prev_map_state, self.last_action, self.dead_reward, new_map_state, True)
        self.replay_memory.append(step)
        self.final_train()
        self.save_weights()
        self.lengths.append(self.length)
        self.total_rewards.append(self.total_reward)
        self.nr_of_steps.append(self.steps)
        return super().kill()

    def reset(self):
        self.steps = 0
        self.total_reward = 0
        return super().reset()
