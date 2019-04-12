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

class DQNKeras(LearningModel):
    def __init__(self, agent, save_path, learning_rate=0.001, gamma=0.99, name='DQN', epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995, n_resets=0, batch_size=32):
        super(DQNKeras, self).__init__(agent, save_path, name)

        self.learning_rate = learning_rate
        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.n_resets = n_resets
        self.batch_size = batch_size

        self.model = self.build_model()
        self.memory = deque(maxlen=50000)
        #self.saver = tf.train.Saver()
        #self.load()


    def build_model(self):
        model = Sequential()
        model.add(Dense(12, activation='relu', input_dim=self.state_size))
        model.add(Dense(12, activation='relu'))
        model.add(Dense(12, activation='relu'))
        model.add(Dense(self.action_size))
        model.compile(Adam(lr=self.learning_rate), 'mse')

        return model
    

    def learn(self, s, a, r, s_, done):
        self.record(s, a, r, s_, done)

        if done:
            self.replay()


    def replay(self):
        if len(self.memory) >= self.batch_size:
            minibatch = random.sample(self.memory, self.batch_size)
            for s, a, r, s_, done in minibatch:
                target = r

                if not done:
                    target = (r + self.gamma * np.amax(self.model.predict(s_)[0]))
                
                target_f = self.model.predict(s)
                target_f[0][np.argmax(a)] = target
                self.model.fit(s, target_f, epochs=1, verbose=0)

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    
    def record(self, s, a, r, s_, done):
        self.memory.append((s, a, r, s_, done))


    def choose_action(self, state, excluded_actions=[], is_playing=False):
        if np.random.rand() <= self.epsilon:
            action = random.choice(self.actions)
            return action
        return self.predict(state)

    def predict(self, state):
        '''
        model.predict returns an array of arrays, containing the Q-Values for the actions. This function should return the
        corresponding action with the highest Q-Value.
        '''
        return self.actions[int(np.argmax(self.model.predict(state)[0]))]


    def save(self):
        print()
        print("> Save not yet implemented!")
        print()

    def load(self):
        print()
        print("> Load not yet implemented!")
        print()
