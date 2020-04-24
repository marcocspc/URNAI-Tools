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
from .base.abmodel import LearningModel
from agents.actions.base.abwrapper import ActionWrapper
from agents.states.abstate import StateBuilder

class DQNKerasMem(LearningModel):

    def __init__(self, action_wrapper: ActionWrapper, state_builder: StateBuilder, learning_rate=0.002, gamma=0.95, 
                name='DQN', epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995, n_resets=0, batch_size=32,
                nodes_layer1=200, nodes_layer2=200, nodes_layer3=200, nodes_layer4=200, memory_maxlen=2000):
        super(DQNKerasMem, self).__init__(action_wrapper, state_builder, gamma, learning_rate, name)

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.n_resets = n_resets
        self.batch_size = batch_size

        self.nodes_layer1 = nodes_layer1
        self.nodes_layer2 = nodes_layer2
        self.nodes_layer3 = nodes_layer3
        self.nodes_layer4 = nodes_layer4
        self.memory_maxlen = memory_maxlen

        self.state_size = int(self.state_size)

        self.model = self.build_model()
        self.memory = deque(maxlen=self.memory_maxlen)
        
    def build_model(self):
        model = Sequential()
        model.add(Dense(self.nodes_layer1, input_dim=self.state_size, activation='relu'))
        model.add(Dense(self.nodes_layer2, activation='relu'))
        model.add(Dense(self.nodes_layer3, activation='relu'))
        model.add(Dense(self.nodes_layer4, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        return model

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
                
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def learn(self, s, a, r, s_, done, is_last_step: bool):
        self.memorize(s, a, r, s_, done)
        if(len(self.memory) > self.batch_size):
            self.replay()

    def choose_action(self, state, excluded_actions=[]):
        if np.random.rand() <= self.epsilon:
            random_action = random.choice(self.actions)
            # Removing excluded actions
            while random_action in excluded_actions:
                random_action = random.choice(self.actions)
            return random_action
        return self.predict(state)

    def predict(self, state):
        '''
        model.predict returns an array of arrays, containing the Q-Values for the actions. This function should return the
        corresponding action with the highest Q-Value.
        '''
        return self.actions[int(np.argmax(self.model.predict(state)[0]))]

    def save_extra(self, persist_path):
        self.model.save_weights(self.get_full_persistance_path(persist_path)+".h5")

    def load_extra(self, persist_path):
        exists = os.path.isfile(self.get_full_persistance_path(persist_path)+".h5")

        if(exists):
            build_model()
            self.model.load_weights(self.get_full_persistance_path(persist_path)+".h5")

