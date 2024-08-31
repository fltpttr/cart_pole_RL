"""Nature (D)DQN agent."""
import os
import keras.optimizers
import numpy as np
from keras import Sequential
from keras.layers import Input, Dense
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class DQN:
    def __init__(self, samples_num=3000, epsilon=1, gamma=0.95, epsilon_step=0.001, epsilon_x=1, max_step=6000,
                 hidden1=64, hidden2=64, rms_prop_step=0.0005, batch_size=32, model_target_update_step=15,
                 double_q=True):
        self.epsilon = epsilon
        self.epsilon_x = epsilon_x
        self.epsilon_step = epsilon_step
        self.gamma = gamma
        self.samples_num = samples_num
        self.batch_size = batch_size
        self.step_counter = 0
        self.max_step = max_step
        self.action = 0

        # Reply buffer.
        self.states = np.zeros(shape=(samples_num, 4), dtype='float32')
        self.actions = np.zeros(shape=samples_num, dtype='uint8')
        self.rewards = np.zeros(shape=samples_num, dtype='float32')
        self.terms = np.zeros(shape=samples_num, dtype='bool')
        self.td_targets = np.zeros(shape=(samples_num, 2), dtype='float32')

        self.model = Sequential([
            Dense(hidden1, activation='relu', input_shape=(4,)),
            Dense(hidden2, activation='relu'),
            Dense(2, activation='linear')
        ])

        self.model_target = Sequential([
            Dense(hidden1, activation='relu', input_shape=(4,)),
            Dense(hidden2, activation='relu'),
            Dense(2, activation='linear')
        ])

        self.model.compile(loss='mse', optimizer=keras.optimizers.RMSprop(learning_rate=rms_prop_step))
        self.model_target.compile(loss='mse', optimizer=keras.optimizers.RMSprop(learning_rate=rms_prop_step))
        self.model_target.set_weights(self.model.get_weights())
        self.model_target_update_step = model_target_update_step
        self.double_q = double_q

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

    def choosing_action(self):
        # Epsilon-greedy strategy.
        if np.random.rand() > self.epsilon and self.step_counter > self.samples_num:
            self.action = np.argmax(self.model.predict(np.expand_dims(self.states[-1], axis=0), verbose=0)[0])
        else:
            if self.step_counter % 5 == 0:
                self.action = np.random.randint(0, 2)

        # Save action.
        if self.step_counter < self.samples_num:
            self.actions[self.step_counter] = self.action
        else:
            self.actions = np.roll(self.actions, -1, axis=0)
            self.actions[-1] = self.action
        return self.action

    def save_reward(self, reward):
        # If the experience array has not been fully filled yet.
        if self.step_counter < self.samples_num:
            # If the previous state was terminal, reset the reward.
            if self.step_counter > 0 and self.terms[self.step_counter]:
                reward = 0
            self.rewards[self.step_counter] = reward
        else:
            # If the previous state was terminal, reset the reward.
            if self.terms[-1]:
                reward = 0
            self.rewards = np.roll(self.rewards, -1, axis=0)
            self.rewards[-1] = reward

    def step_increment(self):
        self.step_counter += 1

    def epsilon_decrement(self, score, max_score):
        self.epsilon = (np.sin(self.epsilon_x) + 1) / 2
        self.epsilon_x += self.epsilon_step
        if score > 650:
            self.max_step = self.step_counter - 1
            self.epsilon = 0
        if self.max_step == self.step_counter and max_score < 650:
            self.max_step += 1000

    def train_step(self):
        # Take random batch.
        random_indxs = np.random.choice(np.arange(0, self.samples_num - 1), size=self.batch_size, replace=False)

        # Q - values.
        self.td_targets[random_indxs] = self.model_target.predict(self.states[random_indxs], verbose=0)

        # Double Q-learning or not.
        if self.double_q:
            actions_indxs = np.argmax(self.model(self.states[random_indxs + 1]), axis=1)
            self.td_targets[random_indxs, self.actions[random_indxs]] = self.rewards[random_indxs] + self.gamma * \
                 self.model_target(self.states[random_indxs + 1]).numpy()[np.arange(self.batch_size), actions_indxs] * \
                                  (1 - self.terms[random_indxs + 1])
        else:
            self.td_targets[random_indxs, self.actions[random_indxs]] = self.rewards[random_indxs] + \
                self.gamma * np.max(self.model_target.predict(self.states[random_indxs + 1], verbose=0), axis=1) * \
                (1 - self.terms[random_indxs + 1])

        self.td_targets[random_indxs] *= np.expand_dims(1 - self.terms[random_indxs], axis=1)

        # Q - learning.
        self.model.fit(self.states[random_indxs], self.td_targets[random_indxs], batch_size=self.batch_size,
                                                                                                        verbose=0)
        # Update the target model.
        if self.step_counter % self.model_target_update_step == 0:
            self.model_target.set_weights(self.model.get_weights())
