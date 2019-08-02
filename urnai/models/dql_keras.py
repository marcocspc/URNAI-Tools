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

class DQNKeras(LearningModel):
    def __init__(self, action_wrapper: ActionWrapper, state_builder: StateBuilder, save_path, learning_rate=0.001, gamma=0.95,
                    name='DQN', epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995, n_resets=0, batch_size=32, memory_size=50000):
        super(DQNKeras, self).__init__(action_wrapper, state_builder, gamma, learning_rate, save_path, name)

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.n_resets = n_resets
        self.batch_size = batch_size

        self.model = self.__build_model()
        self.memory = deque(maxlen=memory_size)
        self.load()


    def __build_model(self):
        model = Sequential()
        model.add(Dense(12, activation='relu', input_dim=self.state_size))
        model.add(Dense(12, activation='relu'))
        model.add(Dense(12, activation='relu'))
        model.add(Dense(self.action_size))
        model.compile(Adam(lr=self.learning_rate), 'mse')

        return model
    

    def learn(self, s, a, r, s_, done, is_last_step):
        self.__record(s, a, r, s_, done)

        # self.__replay()

        if is_last_step or done:
            self.__replay()


    def __replay(self):
        if len(self.memory) < self.batch_size:
            return 0
        
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

    
    def __record(self, s, a, r, s_, done):
        self.memory.append((s, a, r, s_, done))


    def choose_action(self, state, excluded_actions=[]):
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
        print("\n> Saving...\n")
        self.model.save_weights(self.save_path + '.h5')
        print("> The model's weights were succesfully saved!\n")


    def load(self):
        print("\n> Loading...\n")
        exists = os.path.isfile(self.save_path + '.h5')
        if exists:
            self.model.load_weights(self.save_path + '.h5')
            print("> The model's weights were succesfully loaded!\n")
        else:
            print("> There are no weights saved in " + self.save_path)
