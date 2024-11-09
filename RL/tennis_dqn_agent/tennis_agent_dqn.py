import numpy as np
import tensorflow as tf


class DQN:
    def __init__(self, samples_num=3000, epsilon=1, min_epsilon=0.2, epsilon_step=0.00005, gamma=0.95, hidden1=64,
                 hidden2=64, rms_step=0.0005, batch_size=64, double_q=False):
        self.double_q = double_q

        self.samples_num = samples_num
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_step = epsilon_step
        self.gamma = gamma
        self.batch_size = batch_size

        self.action = -1
        self.reward = 0
        self.step_counter = 0

        # Reply buffer.
        self.states = np.zeros(shape=(self.samples_num, 6), dtype='float32')
        self.terms = np.zeros(shape=self.samples_num, dtype='bool')
        self.actions = np.zeros(shape=self.samples_num, dtype='int8') - 1
        self.rewards = np.zeros(shape=self.samples_num, dtype='int8')
        self.td_targets = np.zeros(shape=(self.samples_num, 2), dtype='float32')

        # Model.
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(hidden1, 'relu', input_shape=(6,)),
            tf.keras.layers.Dense(hidden2, 'relu'),
            tf.keras.layers.Dense(2, 'linear')
        ])
        self.model.compile(loss='mse',
                           optimizer=tf.optimizers.RMSprop(learning_rate=rms_step))

        self.target_model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(hidden1, 'relu', input_shape=(6,)),
            tf.keras.layers.Dense(hidden2, 'relu'),
            tf.keras.layers.Dense(2, 'linear')
        ])
        self.target_model.compile(loss='mse',
                                  optimizer=tf.optimizers.RMSprop(learning_rate=rms_step))
        self.target_model.set_weights(self.model.get_weights())

    def choose_action(self, state):
        if np.random.rand() > self.epsilon:
            self.action = np.argmax(self.model.predict(np.expand_dims(state, axis=0), verbose=0))
        else:
            if self.step_counter % np.random.randint(2, 10) == 0:
                self.action = np.random.choice([0, 1])
        if self.step_counter > self.samples_num:
            self.epsilon -= self.epsilon_step
            if self.epsilon < self.min_epsilon:
                self.epsilon = 0
        return self.action

    def save_state(self, state, term):
        # If the experience array has not been fully filled yet.
        if self.step_counter < self.samples_num - 1:
            self.states[self.step_counter] = state
            self.actions[self.step_counter] = self.action
            self.terms[self.step_counter] = term
        else:
            self.states = np.roll(self.states, -1, axis=0)
            self.states[-1] = state
            self.actions = np.roll(self.actions, -1, axis=0)
            self.actions[-1] = self.action
            self.terms = np.roll(self.terms, -1, axis=0)
            self.terms[-1] = term
            self.td_targets = np.roll(self.td_targets, -1, axis=0)

    def save_reward(self, reward):
        # If the experience array has not been fully filled yet.
        if self.step_counter < self.samples_num - 1:
            self.rewards[self.step_counter] = reward
        else:
            self.rewards = np.roll(self.rewards, -1, axis=0)
            self.rewards[-1] = reward

    def train_model(self):
        if self.step_counter > self.samples_num and self.epsilon > self.min_epsilon:
            random_indxs = np.random.choice(np.arange(self.samples_num - 1), size=self.batch_size, replace=False)

            # target_model?
            self.td_targets[random_indxs] = self.target_model(self.states[random_indxs]) * \
                        (1 - np.vstack((self.terms[random_indxs], self.terms[random_indxs])).T)

            if self.double_q:
                next_actions_argmax = np.argmax(self.model(self.states[random_indxs + 1]), axis=1)
                self.td_targets[random_indxs, self.actions[random_indxs]] = self.rewards[random_indxs] + self.gamma * \
                    self.target_model(self.states[random_indxs + 1]).numpy()[np.arange(self.batch_size),
                        next_actions_argmax] * (1 - self.terms[random_indxs + 1])
            else:
                self.td_targets[random_indxs, self.actions[random_indxs]] = self.rewards[random_indxs] + self.gamma * \
                    np.max(self.target_model(self.states[random_indxs + 1]), axis=1) * \
                    (1 - self.terms[random_indxs + 1])

            self.model.fit(self.states[random_indxs], self.td_targets[random_indxs],
                           batch_size=self.batch_size, verbose=0)

            if self.step_counter % 15 == 0:
                self.target_model.set_weights(self.model.get_weights())

    def save_model(self, s):
        tf.keras.models.save_model(self.model, '.\\saved_models\\tennis_vanilla_dqn_' + s)

    def step_counter_increment(self):
        self.step_counter += 1
