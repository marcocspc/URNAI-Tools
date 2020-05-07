import tensorflow as tf
import numpy as np
import random
import os
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten 
from keras.optimizers import Adam
from keras import backend as K
from .base.abmodel import LearningModel
from agents.actions.base.abwrapper import ActionWrapper
from agents.states.abstate import StateBuilder
from .model_builder import ModelBuilder
from .dql_keras_mem import DQNKerasMem 

class DDQNKeras(DQNKerasMem):

    def __init__(self, action_wrapper: ActionWrapper, state_builder: StateBuilder, learning_rate=0.002, gamma=0.95, 
                name='DQN', epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995, n_resets=0, batch_size=32,
                memory_maxlen=2000, use_memory=True, per_episode_epsilon_decay=False, build_model = ModelBuilder.DEFAULT_BUILD_MODEL):
        super(DDQNKeras, self).__init__(action_wrapper, state_builder, learning_rate, gamma, name, epsilon, epsilon_min, epsilon_decay, n_resets, batch_size, memory_maxlen, use_memory, per_episode_epsilon_decay, build_model)

        self.model = self.make_model()
        self.target_model = self.make_model()

    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond  = K.abs(error) <= clip_delta

        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

        return K.mean(tf.where(cond, squared_loss, quadratic_loss))
        
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def replay(self):
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * np.amax(t)
            self.model.fit(state, target, epochs=1, verbose=0)
        #Epsilon decay operation was here, moved it to "decay_epsilon()" and to "learn()"

    def no_memory_learning(self, s, a, r, s_, done, is_last_step):
        target = self.model.predict(s)
        if done:
            target[0][a] = r 
        else:
            t = self.target_model.predict(s_)[0]
            target[0][a] = r + self.gamma * np.amax(t)
        self.model.fit(s, target, epochs=1, verbose=0)

    def learn(self, s, a, r, s_, done, is_last_step: bool):
        if self.use_memory:
            self.memorize(s, a, r, s_, done)
            if(len(self.memory) > self.batch_size):
                self.replay()
            if(done):
                self.update_target_model()
        else:
            #TODO test learning without memory:
            self.no_memory_learning(s, a, r, s_, done, is_last_step)

        if not self.per_episode_epsilon_decay:
            self.decay_epsilon()

    def predict(self, state, excluded_actions=[]):
        '''
        model.predict returns an array of arrays, containing the Q-Values for the actions. This function should return the
        corresponding action with the highest Q-Value.
        '''
        return self.actions[int(np.argmax(self.model.predict(state)[0]))]

    def save_extra(self, persist_path):
        self.model.save_weights(self.get_full_persistance_path(persist_path)+"_model_"+".h5")
        self.target_model.save_weights(self.get_full_persistance_path(persist_path)+"_target_model_"+".h5")

    def load_extra(self, persist_path):
        exists_model = os.path.isfile(self.get_full_persistance_path(persist_path)+"_model_"+".h5")
        exists_target = os.path.isfile(self.get_full_persistance_path(persist_path)+"_target_model_"+".h5")

        if(exists_model and exists_target):
            self.model = self.make_model()
            self.target_model = self.make_model()
            self.model.load_weights(self.get_full_persistance_path(persist_path)+"_model_"+".h5")
            self.target_model.load_weights(self.get_full_persistance_path(persist_path)+"_target_model_"+".h5")
