"""Dueling (D)DQN agent + Polyak averaging + PER."""
import os
import tensorflow as tf
import keras.optimizers
import numpy as np
from keras import Model
from keras.layers import Input, Dense
from keras.losses import Huber
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class DuelingDeepQNetwork(Model):
    def __init__(self, hidden1_units=64, hidden2_units=64):
        super(DuelingDeepQNetwork, self).__init__()
        self.dense1 = Dense(hidden1_units, activation='relu', input_shape=(4,))
        self.dense2 = Dense(hidden2_units, activation='relu')
        self.V = Dense(1, activation=None)
        self.A = Dense(2, activation=None)

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        V = self.V(x)
        A = self.A(x)
        Q = (V + (A - tf.reduce_mean(A, axis=1, keepdims=True)))
        return Q

    def advantage(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        A = self.A(x)
        return A


class DuelingPerDQN:
    def __init__(self, samples_num=1000, gamma=0.95, hidden1=64, hidden2=64, epsilon=1, epsilon_step=0.00002,
                 rms_prop_step=0.0007, batch_size=64, double_q=True, polyak_averaging=True, tau=0.1, per=True):
        self.epsilon = epsilon
        self.epsilon_step = epsilon_step
        self.gamma = gamma
        self.samples_num = samples_num
        self.batch_size = batch_size
        self.step_counter = 0
        self.action = 0
        self.states = np.zeros(shape=(samples_num, 4), dtype='float32')
        self.actions = np.zeros(shape=samples_num, dtype='uint8')
        self.rewards = np.zeros(shape=samples_num, dtype='float32')
        self.terms = np.zeros(shape=samples_num, dtype='bool')
        self.td_targets = np.zeros(shape=(samples_num, 2), dtype='float32')
        self.polyak_averaging = polyak_averaging

        self.per = per
        self.sample_priorities = np.zeros(shape=samples_num, dtype='float32') + 0.000001
        self.sample_probabilities = np.zeros(shape=samples_num, dtype='float32')
        self.sample_weights = np.ones(shape=self.samples_num)
        self.alpha = 0.6
        self.beta = 0.1
        self.beta_step = 0.00002

        self.model = DuelingDeepQNetwork(hidden1, hidden2)
        self.target_model = DuelingDeepQNetwork(hidden1, hidden2)
        self.model.compile(loss='mse', optimizer=keras.optimizers.RMSprop(learning_rate=rms_prop_step))
        self.target_model.compile(loss='mse', optimizer=keras.optimizers.RMSprop(learning_rate=rms_prop_step))
        self.target_model.set_weights(self.model.get_weights())
        self.double_q = double_q
        self.tau = tau

    def save_state(self, state_sample, term):
        # If the experience array has not been fully filled yet.
        if self.step_counter < self.samples_num:
            self.states[self.step_counter] = state_sample
            self.terms[self.step_counter] = term
            if self.step_counter > 0 and self.terms[self.step_counter - 1]:
                self.states[self.step_counter][1] = 0
                self.states[self.step_counter][3] = 0
        else:
            self.states = np.roll(self.states, -1, axis=0)
            self.states[-1] = state_sample
            self.terms = np.roll(self.terms, -1, axis=0)
            self.terms[-1] = term

            # If the previous state was terminal.
            if self.terms[-2]:
                self.states[-1][1] = 0
                self.states[-1][3] = 0
            self.td_targets = np.roll(self.td_targets, -1, axis=0)

    def epsilon_decrement(self):
        self.epsilon -= self.epsilon_step

    def choosing_action(self):
        # Epsilon-greedy strategy.
        if np.random.rand() > self.epsilon and self.step_counter > self.samples_num:
            self.action = np.argmax(self.model.advantage(np.expand_dims(self.states[-1], axis=0)), axis=1)[0]
        else:
            if self.step_counter % 5 == 0:
                self.action = np.random.randint(0, 2)

        if self.step_counter > self.samples_num:
            self.epsilon_decrement()

        # Save action.
        if self.step_counter < self.samples_num:
            self.actions[self.step_counter] = self.action
        else:
            self.actions = np.roll(self.actions, -1, axis=0)
            self.actions[-1] = self.action
        return self.action

    def save_reward(self, reward):
        # If the experience array has not been fully filled yet.
        if self.step_counter < self.samples_num - 1:
            # If the previous state was terminal, reset the reward.
            if self.step_counter > 0 and self.terms[self.step_counter]:
                reward = 0
            self.rewards[self.step_counter] = reward

            if self.per and self.step_counter > 0:
                td_target = self.rewards[self.step_counter-1] + self.gamma * np.max(self.target_model(np.expand_dims(
                    self.states[self.step_counter], axis=0)), axis=1) * (1 - self.terms[self.step_counter])
                td_error = td_target - np.max(self.model(np.expand_dims(self.states[self.step_counter-1], axis=0)),
                                              axis=1)
                self.sample_priorities[self.step_counter-1] = np.abs(td_error) + 0.000001
        else:
            # If the previous state was terminal, reset the reward.
            if self.terms[-1]:
                reward = 0
            self.rewards = np.roll(self.rewards, -1, axis=0)
            self.rewards[-1] = reward

            if self.per:
                td_target = self.rewards[-2] + self.gamma * np.max(self.target_model(np.expand_dims(
                    self.states[-1], axis=0)), axis=1) * (1 - self.terms[-1])
                td_error = td_target - np.max(self.model(np.expand_dims(self.states[-2], axis=0)), axis=1)
                self.sample_priorities = np.roll(self.sample_priorities, -1, axis=0)
                self.sample_priorities[-1] = 0.000001
                self.sample_priorities[-2] = np.abs(td_error) + 0.000001
                self.sample_probabilities = np.power(self.sample_priorities, self.alpha) / np.sum(np.power(
                    self.sample_priorities, self.alpha))
                self.sample_weights = np.power(self.samples_num * self.sample_probabilities, -self.beta)
                self.sample_weights[-1] = 0
                self.sample_weights = self.sample_weights / np.max(self.sample_weights)
                if self.beta < 1 - self.beta_step:
                    self.beta += self.beta_step

    def step_increment(self):
        self.step_counter += 1

    def train_step(self):
        if self.step_counter > self.samples_num and self.epsilon > 0.2:
            # Take random batch. PER or not.
            if self.per:
                random_indxs = np.random.choice(np.arange(self.samples_num - 1), size=self.batch_size, replace=False,
                                                p=self.sample_probabilities[:-1])
            else:
                random_indxs = np.random.choice(np.arange(self.samples_num - 1), size=self.batch_size, replace=False)

            # Q - values.
            self.td_targets[random_indxs] = self.target_model(self.states[random_indxs]) * (1 - np.vstack(
                (self.terms[random_indxs], self.terms[random_indxs])).T)

            # Double Q-learning or not.
            if self.double_q:
                actions_indxs = np.argmax(self.model(self.states[random_indxs + 1]), axis=1)
                self.td_targets[random_indxs, self.actions[random_indxs]] = self.rewards[random_indxs] + self.gamma * \
                    self.target_model(self.states[random_indxs + 1]).numpy()[np.arange(
                        self.batch_size), actions_indxs] * (1 - self.terms[random_indxs + 1])
            else:
                self.td_targets[random_indxs, self.actions[random_indxs]] = self.rewards[random_indxs] + \
                    self.gamma * np.max(self.target_model(self.states[random_indxs + 1]), axis=1) * \
                        (1 - self.terms[random_indxs + 1])

            # Q - learning.
            self.model.fit(self.states[random_indxs], self.td_targets[random_indxs], batch_size=self.batch_size,
                           sample_weight=self.sample_weights[random_indxs], verbose=0)

            # Update the target model.
            if self.polyak_averaging:
                weights = []
                for i in range(len(self.target_model.layers)):
                    for j in range(2):
                        weights.append(self.tau * self.model.layers[i].get_weights()[j] + (1 - self.tau) * \
                                       self.target_model.layers[i].get_weights()[j])
                self.target_model.set_weights(weights)
            else:
                if self.step_counter % 15 == 0:
                    self.target_model.set_weights(self.model.get_weights())

