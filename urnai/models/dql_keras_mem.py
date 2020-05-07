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
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D 
from keras.optimizers import Adam
from .base.abmodel import LearningModel
from agents.actions.base.abwrapper import ActionWrapper
from agents.states.abstate import StateBuilder
from .model_builder import ModelBuilder
from urnai.utils.error import IncoherentBuildModelError

class DQNKerasMem(LearningModel):

    def __init__(self, action_wrapper: ActionWrapper, state_builder: StateBuilder, learning_rate=0.002, gamma=0.95, 
                name='DQN', epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995, n_resets=0, batch_size=32,
                memory_maxlen=2000, use_memory=True, per_episode_epsilon_decay=False, build_model = ModelBuilder.DEFAULT_BUILD_MODEL):
        super(DQNKerasMem, self).__init__(action_wrapper, state_builder, gamma, learning_rate, epsilon, epsilon_min, epsilon_decay, per_episode_epsilon_decay, name)
        self.n_resets = n_resets
        self.batch_size = batch_size

        self.build_model = build_model
        self.model = self.make_model()
        self.use_memory = use_memory

        if self.use_memory:
            self.memory = deque(maxlen=memory_maxlen)
            self.memory_maxlen = memory_maxlen
        
    def make_model(self):
        model = Sequential()

        if self.build_model[0]['type'] == ModelBuilder.LAYER_INPUT and self.build_model[-1]['type'] == ModelBuilder.LAYER_OUTPUT:
            self.build_model[0]['shape'] = [None, self.state_size]
            self.build_model[-1]['length'] = self.action_size

        for layer_model in self.build_model:
            if layer_model['type'] == ModelBuilder.LAYER_INPUT: 
                if self.build_model.index(layer_model) == 0:
                    model.add(Dense(layer_model['nodes'], input_dim=layer_model['shape'][1], activation='relu'))
                else:
                    raise IncoherentBuildModelError("Input Layer must be the first one.") 
            elif layer_model['type'] == ModelBuilder.LAYER_FULLY_CONNECTED:
                idx = self.build_model.index(layer_model) - 1 
                if self.build_model[idx]['type'] == ModelBuilder.LAYER_CONVOLUTIONAL:
                    model.add(Flatten())

                model.add(Dense(layer_model['nodes'], activation='relu'))
            elif layer_model['type'] == ModelBuilder.LAYER_OUTPUT:
                idx = self.build_model.index(layer_model) - 1 
                if self.build_model[idx]['type'] == ModelBuilder.LAYER_CONVOLUTIONAL:
                    model.add(Flatten())

                model.add(Dense(layer_model['length'], activation='linear'))
            elif layer_model['type'] == ModelBuilder.LAYER_CONVOLUTIONAL:
                if self.build_model.index(layer_model) == 0:
                    model.add(Conv2D(layer_model['filters'], layer_model['filter_shape'], 
                              padding=layer_model['padding'], activation='relu', input_shape=layer_model['input_shape']))
                    model.add(MaxPooling2D(pool_size=layer_model['max_pooling_pool_size_shape']))
                else:
                    model.add(Conv2D(layer_model['filters'], layer_model['filter_shape'], 
                              padding=layer_model['padding'], activation='relu'))
                    model.add(MaxPooling2D(pool_size=layer_model['max_pooling_pool_size_shape']))
            else:
                raise UnsupportedBuildModelLayerTypeError("Unsuported Layer Type " + layer_model['type'])


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
        #Epsilon decay operation was here, moved it to "decay_epsilon()" and to "learn()"

    def no_memory_learning(self, s, a, r, s_, done, is_last_step):
            target = r 
            if not done:
                target = (r + self.gamma * np.amax(self.model.predict(s_)[0]))
            target_f = self.model.predict(s)
            target_f[0][a] = target
            self.model.fit(s, target_f, epochs=1, verbose=0)

    def learn(self, s, a, r, s_, done, is_last_step: bool):
        if self.use_memory:
            self.memorize(s, a, r, s_, done)
            if(len(self.memory) > self.batch_size):
                self.replay()
        else:
            self.no_memory_learning(s, a, r, s_, done, is_last_step)

        if not self.per_episode_epsilon_decay:
            self.decay_epsilon()


    def choose_action(self, state, excluded_actions=[]):
        if np.random.rand() <= self.epsilon_greedy:
            random_action = random.choice(self.actions)
            # Removing excluded actions
            while random_action in excluded_actions:
                random_action = random.choice(self.actions)
            return random_action
        else:
            return self.predict(state)

    def save_extra(self, persist_path):
        self.model.save_weights(self.get_full_persistance_path(persist_path)+".h5")

    def load_extra(self, persist_path):
        exists = os.path.isfile(self.get_full_persistance_path(persist_path)+".h5")

        if(exists):
            self.model = self.make_model()
            self.model.load_weights(self.get_full_persistance_path(persist_path)+".h5")

    def predict(self, state, excluded_actions=[]):
        '''
        model.predict returns an array of arrays, containing the Q-Values for the actions. This function should return the
        corresponding action with the highest Q-Value.
        '''
        return self.actions[int(np.argmax(self.model.predict(state)[0]))]

