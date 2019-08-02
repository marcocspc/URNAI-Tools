'''
This is the exact same model as our DQNKeras class, except that this one has a memory system in place:

During training the model will select an action (either randomly or through the weights)
and then store the state, action, reward and previous state tuple in the memory (s, a, r, s_).

Right after that the model will randomly sample a batch from the memory and train from that.

Remember that the model's memory has a limited size, to avoid performance issues and to make it so
that the model always trains with relatively recent memories.
'''

import tensorflow as tf
import numpy as np
import random
import os
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from .dql_keras import DQNKeras
from agents.actions.base.abwrapper import ActionWrapper
from agents.states.abstate import StateBuilder

class DQNKerasMem(DQNKeras):

    def __init__(self, action_wrapper: ActionWrapper, state_builder: StateBuilder, save_path, learning_rate=0.0002, gamma=0.95, name='DQN', epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995, n_resets=0, batch_size=32):
        super(DQNKerasMem, self).__init__(action_wrapper, state_builder, gamma, learning_rate, save_path, name)

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.n_resets = n_resets
        self.batch_size = batch_size

        self.model = self.build_model()
        self.memory = deque(maxlen=5000)
        self.load()
        
    def build_model(self):
        model = Sequential()
        model.add(Dense(16, activation='relu', input_dim=self.state_size))
        model.add(Dense(16, activation='relu'))
        #model.add(Dense(12, activation='relu'))
        model.add(Dense(self.action_size))
        model.compile(Adam(lr=self.learning_rate), 'mse')

        return model

    def learn(self, s, a, r, s_, done):
        self.__record(s, a, r, s_, done)
        self.replay()

    def replay(self):
        if len(self.memory) >= self.batch_size:
            minibatch = random.sample(self.memory, self.batch_size)
            for s, a, r, s_, done in minibatch:
                target = r

                if not done:
                    target = (r + self.gamma * np.amax(self.model.predict(s_)[0]))
                
                target_f = self.model.predict(s)
                target_f[0][a] = target
                self.model.fit(s, target_f, epochs=1, verbose=0)

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
