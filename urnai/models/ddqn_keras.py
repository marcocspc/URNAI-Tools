import tensorflow as tf
import numpy as np
import random
import os
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Activation
from keras.optimizers import Adam
from keras import backend as K
from .base.abmodel import LearningModel
from agents.actions.base.abwrapper import ActionWrapper
from agents.states.abstate import StateBuilder
from .model_builder import ModelBuilder
from urnai.utils.error import IncoherentBuildModelError
from urnai.utils.error import UnsupportedBuildModelLayerTypeError

class DDQNKeras(LearningModel):
    def __init__(self, action_wrapper: ActionWrapper, state_builder: StateBuilder, learning_rate=0.001, gamma=0.99,
                    name='DDQN', epsilon_start=1.0, epsilon_min=0.1, epsilon_decay=0.995, per_episode_epsilon_decay=False, update_target_every=5, 
                    batch_size=64, memory_maxlen=50000, min_memory_size=1000, build_model = ModelBuilder.DEFAULT_BUILD_MODEL):
        super(DDQNKeras, self).__init__(action_wrapper, state_builder, gamma, learning_rate, epsilon_start, epsilon_min, epsilon_decay, per_episode_epsilon_decay, name)

        self.build_model = build_model

        # Main model, trained every step
        self.model = self.make_model()
        # Target model, used in .predict every step (does not get update every step)
        self.target_model = self.make_model()
        self.target_model.set_weights(self.model.get_weights())
        self.target_update_counter = 0
        self.update_target_every = update_target_every

        self.memory = deque(maxlen=memory_maxlen)
        self.memory_maxlen = memory_maxlen
        self.min_memory_size = min_memory_size
        self.batch_size = batch_size

    def make_model(self):
        model = Sequential()

        if self.build_model[0]['type'] == ModelBuilder.LAYER_INPUT and self.build_model[-1]['type'] == ModelBuilder.LAYER_OUTPUT:
            self.build_model[0]['shape'] = [None, self.state_size]
            self.build_model[-1]['length'] = self.action_size

        for idx, (layer_model) in enumerate(self.build_model):
            if layer_model['type'] == ModelBuilder.LAYER_INPUT: 
                if self.build_model.index(layer_model) == 0:
                    model.add(Dense(layer_model['nodes'], input_dim=layer_model['shape'][1], activation='relu'))
                else:
                    raise IncoherentBuildModelError("Input Layer must be the first one.") 

            elif layer_model['type'] == ModelBuilder.LAYER_FULLY_CONNECTED:
                # if previous layer is convolutional, add a Flatten layer before the fully connected
                if self.build_model[idx]['type'] == ModelBuilder.LAYER_CONVOLUTIONAL:
                    model.add(Flatten())

                model.add(Dense(layer_model['nodes'], activation='relu'))

            elif layer_model['type'] == ModelBuilder.LAYER_OUTPUT:
                # if previous layer is convolutional, add a Flatten layer before the fully connected
                if self.build_model[idx]['type'] == ModelBuilder.LAYER_CONVOLUTIONAL:
                    model.add(Flatten())

                model.add(Dense(layer_model['length'], activation='linear'))

            elif layer_model['type'] == ModelBuilder.LAYER_CONVOLUTIONAL:
                # if convolutional layer is the first, it's going to have the input shape and be treated as the input layer
                if self.build_model.index(layer_model) == 0:
                    model.add(Conv2D(layer_model['filters'], layer_model['filter_shape'], 
                              padding=layer_model['padding'], activation='relu', input_shape=layer_model['input_shape']))
                    model.add(Activation('relu'))
                    model.add(MaxPooling2D(pool_size=layer_model['max_pooling_pool_size_shape']))
                    model.add(Dropout(0.2))
                else:
                    model.add(Conv2D(layer_model['filters'], layer_model['filter_shape'], 
                              padding=layer_model['padding'], activation='relu'))
                    model.add(Activation('relu'))
                    model.add(MaxPooling2D(pool_size=layer_model['max_pooling_pool_size_shape']))
                    model.add(Dropout(0.2))
            else:
                raise UnsupportedBuildModelLayerTypeError("Unsuported Layer Type " + layer_model['type'])

        model.compile(optimizer=Adam(lr=self.learning_rate), loss='mse', metrics=['accuracy'])
        return model

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def learn(self, s, a, r, s_, done, is_last_step):
        self.memorize(s, a, r, s_, done)
        if len(self.memory) < self.min_memory_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)

        # array of initial states from the minibatch
        current_states = np.array([transition[0] for transition in minibatch])
        # removing undesirable dimension created by np.array
        current_states = np.squeeze(current_states)
        # array of Q-Values for our initial states
        current_qs_list = self.model.predict(current_states)

        # array of states after step from the minibatch
        next_current_states = np.array([transition[3] for transition in minibatch])
        next_current_states = np.squeeze(next_current_states)
        # array of Q-values for our next states
        next_qs_list = self.target_model.predict(next_current_states)

        # inputs is going to be filled with all current states from the minibatch
        # targets is going to be filled with all of our outputs (Q-Values for each action)
        inputs = []
        targets = []

        for index, (state, action, reward, next_state, done) in enumerate(minibatch):
            # if this step is not the last, we calculate the new Q-Value based on the next_state
            if not done:
                max_next_q = np.max(next_qs_list[index])
                # new Q-value is equal to the reward at that step + discount factor * the max q-value for the next_state
                new_q = reward + self.gamma * max_next_q
            else:
                # if this is the last step, there is no future max q value, so we the new_q is just the reward
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            inputs.append(state)
            targets.append(current_qs)

        np_inputs = np.squeeze(np.array(inputs))
        np_targets = np.array(targets)

        self.model.fit(np_inputs, np_targets, batch_size=self.batch_size, verbose=0, shuffle=False)

        # If it's the end of an episode, increase the target update counter
        if done:
            self.target_update_counter += 1

        # If our target update counter is greater than update_target_every we will update the weights in our target model
        if self.target_update_counter > self.update_target_every:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

        # if our epsilon rate decay is set to be done every step, we simply decay it. Otherwise, this will only be done
        # at the end of every episode, on self.ep_reset() which is in our LearningModel base class
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
    
    def predict(self, state, excluded_actions=[]):
        '''
        model.predict returns an array of arrays, containing the Q-Values for the actions. This function should return the
        corresponding action with the highest Q-Value.
        '''
        return int(np.argmax(self.model.predict(state)[0]))

    def save_extra(self, persist_path):
        self.model.save_weights(self.get_full_persistance_path(persist_path)+".h5")

    def load_extra(self, persist_path):
        exists = os.path.isfile(self.get_full_persistance_path(persist_path)+".h5")

        if(exists):
            self.model = self.make_model()
            self.model.load_weights(self.get_full_persistance_path(persist_path)+".h5")
            self.target_model = self.make_model()
            self.target_model.set_weights(self.model.get_weights())
