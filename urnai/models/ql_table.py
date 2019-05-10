from .base.abmodel import LearningModel
from agents.actions.base.abwrapper import ActionWrapper
from agents.states.abstate import State
import numpy as np


class QLearning(LearningModel):

    def __init__(self, action_wrapper: ActionWrapper, state_builder: State, save_path: str, learning_rate=0.1,
                 gamma=0.95, epsilon=1, epsilon_min=0.01, epsilon_decay=0.996, name='Q-learning'):
        super(QLearning, self).__init__(action_wrapper, state_builder, save_path, name)
        self._setHyperParameters(learning_rate, gamma, epsilon, epsilon_min, epsilon_decay)
        self._q_table = np.zeros([self.state_size, self.action_size])

    def _setHyperParameters(self, learning_rate, gamma, epsilon, epsilon_min, epsilon_decay):
        self._alpha = learning_rate
        self._gamma = gamma
        self._epsilon = epsilon
        self._epsilon_min = epsilon_min
        self._epsilon_decay = epsilon_decay

    def learn(self, current_state, current_action, reward, next_state, done):

        old_value = self._q_table[current_state, current_action]
        max_action = np.max(self._q_table[next_state])

        new_value = (1 - self._alpha) * old_value + self._alpha * (reward + self._gamma * max_action)
        self._q_table[current_state, current_action] = new_value

        if done:
            self._updateEpsilon()

    def _updateEpsilon(self):
        if self._epsilon > self._epsilon_min:
            self._epsilon *= self._epsilon_decay

    def choose_action(self, state, excluded_actions=[]):

        if np.random.uniform(0, 1) < self._epsilon:  # 0.01
            return np.random.choice(self.action_size)
        else:
            return self.predict(state)

    def predict(self, state):
        return np.argmax(self._q_table[state])

    def save(self):
        np.save(self.save_path, self._q_table)

    def load(self):
        self._q_table = np.load(self.save_path)
