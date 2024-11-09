import numpy as np
import tensorflow as tf


class DuelingDQN(tf.keras.models.Model):
    def __init__(self, hidden1, hidden2):
        super().__init__()
        self.hidden1 = tf.keras.layers.Dense(hidden1, 'relu', input_shape=(6,))
        self.hidden2 = tf.keras.layers.Dense(hidden2, 'relu')
        self.V = tf.keras.layers.Dense(1)
        self.A = tf.keras.layers.Dense(2)

    def call(self, state):
        x = self.hidden1(state)
        x = self.hidden2(x)
        V = self.V(x)
        A = self.A(x)
        Q = (V + (A - tf.reduce_mean(A, axis=1, keepdims=True)))
        return Q

    def advantage(self, state):
        x = self.hidden1(state)
        x = self.hidden2(x)
        V = self.V(x)
        A = self.A(x)
        return A

class DQN:
    def __init__(self, samples_num=2000, epsilon=1, min_epsilon=0.2, epsilon_step=0.00005, gamma=0.95, hidden1=64,
                 hidden2=64, rms_step=0.0005, batch_size=64, tau=0.1, alpha=0.6, beta=0.1, beta_step=0.00005,
                 double_q=True, polyak_averaging=True, per=True):
        self.double_q = double_q
        self.polyak_averaging = polyak_averaging
        self.tau = tau
        self.per = per
        self.alpha = alpha
        self.beta = beta
        self.beta_step = beta_step

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
        self.sample_priorities = np.zeros(shape=self.samples_num, dtype='float32') + 0.000001
        self.sample_probabilities = np.zeros(shape=self.samples_num, dtype='float32')
        self.sample_weights = np.ones(shape=self.samples_num)

        # Model.
        self.model = DuelingDQN(hidden1, hidden2)
        self.model.compile(loss='mse',
                           optimizer=tf.optimizers.RMSprop(learning_rate=rms_step))

        self.target_model = DuelingDQN(hidden1, hidden2)
        self.target_model.compile(loss='mse',
                                  optimizer=tf.optimizers.RMSprop(learning_rate=rms_step))
        self.target_model.set_weights(self.model.get_weights())

    def choose_action(self, state):
        if np.random.rand() > self.epsilon:
            self.action = np.argmax(self.model.advantage(np.expand_dims(state, axis=0)))
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

            if self.per and self.step_counter > 0:
                td_target = self.rewards[self.step_counter-1] + self.gamma * np.max(self.target_model(np.expand_dims(
                    self.states[self.step_counter], axis=0)), axis=1) * (1 - self.terms[self.step_counter])
                td_error = td_target - np.max(self.model(np.expand_dims(self.states[self.step_counter-1], axis=0)),
                                              axis=1)
                self.sample_priorities[self.step_counter-1] = np.abs(td_error) + 0.0001
        else:
            self.rewards = np.roll(self.rewards, -1, axis=0)
            self.rewards[-1] = reward

            if self.per:
                td_target = self.rewards[-2] + self.gamma * np.max(self.target_model(np.expand_dims(
                    self.states[-1], axis=0)), axis=1) * (1 - self.terms[-1])
                td_error = td_target - np.max(self.model(np.expand_dims(self.states[-2], axis=0)), axis=1)
                self.sample_priorities = np.roll(self.sample_priorities, -1, axis=0)
                # Последнее значение приоритета не может быть вычислено, так как нельзя получить для него td_error,
                # оно всеравно не будет использоваться, поэтому я просто присвою ему очень малое значение.
                self.sample_priorities[-1] = 0.000001
                self.sample_priorities[-2] = np.abs(td_error) + 0.000001

                # Вероятности и веса для PER на основе приоритетов можно вычислить только если буфер воспроизведения
                # полон.
                self.sample_probabilities = np.power(self.sample_priorities, self.alpha) / np.sum(np.power(
                    self.sample_priorities, self.alpha))
                self.sample_weights = np.power(self.samples_num * self.sample_probabilities, -self.beta)
                self.sample_weights[-1] = 0
                self.sample_weights = self.sample_weights / np.max(self.sample_weights)
                if self.beta < 1 - self.beta_step:
                    self.beta += self.beta_step

    def train_model(self):
        if self.step_counter > self.samples_num and self.epsilon > self.min_epsilon:
            if self.per:
                random_indxs = np.random.choice(np.arange(self.samples_num - 1), size=self.batch_size, replace=False,
                                                p=self.sample_probabilities[:-1])
            else:
                random_indxs = np.random.choice(np.arange(self.samples_num - 1), size=self.batch_size, replace=False)

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
                           batch_size=self.batch_size, sample_weight=self.sample_weights[random_indxs], verbose=0)

            if self.polyak_averaging:
                weights = []
                for i in range(len(self.target_model.layers)):
                    weights.append(self.tau * self.model.layers[i].get_weights()[0] + (1 - self.tau) * \
                                   self.target_model.layers[i].get_weights()[0])
                    weights.append(self.tau * self.model.layers[i].get_weights()[1] + (1 - self.tau) * \
                                   self.target_model.layers[i].get_weights()[1])
                self.target_model.set_weights(weights)
            elif self.step_counter % 15 == 0:
                self.target_model.set_weights(self.model.get_weights())

            # После обучения на выборке состояний, обновить значения их приоритетов.
            # if self.per:
            #     td_errors = self.td_targets[random_indxs, self.actions[random_indxs]] - np.max(self.model(
            #         self.states[random_indxs]), axis=1)
            #     self.sample_priorities[random_indxs] = np.abs(td_errors) + 0.000001

    def save_model(self, s):
        tf.keras.models.save_model(self.model, '.\\saved_models\\tennis_vanilla_dqn_' + s)

    def step_counter_increment(self):
        self.step_counter += 1
