import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import tensorflow as tf
from tensorflow.python.framework import ops
from utils.error import IncoherentBuildModelError, UnsupportedBuildModelLayerTypeError
from tensorflow.compat.v1 import Session,ConfigProto,placeholder,layers,train,global_variables_initializer
import numpy as np
import random
import os
import pickle
from .base.abmodel import LearningModel
from agents.actions.base.abwrapper import ActionWrapper
from agents.states.abstate import StateBuilder
from .model_builder import ModelBuilder

class DqlTfFlexible(LearningModel):

    def __init__(self, action_wrapper: ActionWrapper, state_builder: StateBuilder, learning_rate=0.0002, gamma=0.95, name='DQN', build_model = ModelBuilder.DEFAULT_BUILD_MODEL, epsilon_start=1.0, epsilon_min=0.5, epsilon_decay=0.995, per_episode_epsilon_decay=False):
        super(DqlTfFlexible, self).__init__(action_wrapper, state_builder, gamma, learning_rate, epsilon_start, epsilon_min, epsilon_decay , per_episode_epsilon_decay ,name)
        # Defining the model's layers. Tensorflow's objects are stored into self.model_layers
        self.build_model = build_model
        self.make_model()


    def learn(self, s, a, r, s_, done, is_last_step: bool):
        qsa_values = self.sess.run(self.model_layers[-1], feed_dict={self.model_layers[0]: s})

        current_q = 0

        if done:
            current_q = r
        else:
            current_q = r + self.gamma * self.__maxq(s_)

        qsa_values[0, a] = current_q

        self.sess.run(self.optimizer, feed_dict={self.model_layers[0] : s, self.tf_qsa: qsa_values})

        qsa_values = self.sess.run(self.model_layers[-1], feed_dict={self.model_layers[0]: s})

    def __maxq(self, state):
        values = self.sess.run(self.model_layers[-1], feed_dict={self.model_layers[0]: state})

        index = np.argmax(values[0])
        mxq = values[0, index]

        return mxq

    def choose_action(self, state, excluded_actions=[]):

        expl_expt_tradeoff = np.random.rand()

        if self.epsilon_greedy > expl_expt_tradeoff:
            random_action = random.choice(self.actions)

            # Removing excluded actions
            while random_action in excluded_actions:
                random_action = random.choice(self.actions)
            action = random_action
        else:
            action = self.predict(state, excluded_actions)

        if not self.per_episode_epsilon_decay:
            self.decay_epsilon()

        return action

    def predict(self, state, excluded_actions=[]):
        q_values = self.sess.run(self.model_layers[-1], feed_dict={self.model_layers[0]: state})
        action_idx = np.argmax(q_values)

        # Removing excluded actions
        # This is possibly badly optimized, eventually look back into this
        while action_idx in excluded_actions:
            q_values = np.delete(q_values, action_idx)
            action_idx = np.argmax(q_values)
        
        action = self.actions[int(action_idx)]
        return action

    def save_extra(self, persist_path):
        #Saving tensorflow stuff
        self.saver.save(self.sess, self.get_full_persistance_tensorflow_path(persist_path))

    def load_extra(self, persist_path):
        #Makes model, needed to be done before loading tensorflow's persistance
        self.make_model()
        #Check if tf file exists
        exists = os.path.isfile(self.get_full_persistance_tensorflow_path(persist_path) + ".meta")
        #If yes, load it
        if exists:
            self.saver.restore(self.sess, self.get_full_persistance_tensorflow_path(persist_path))

    def make_model(self):
        #These are already inside make_model(), commenting out
        ops.reset_default_graph()
        tf.compat.v1.disable_eager_execution()

        # Initializing TensorFlow session
        self.sess = Session(config=ConfigProto(allow_soft_placement=True))

        if self.build_model[0]['type'] == ModelBuilder.LAYER_INPUT and self.build_model[-1]['type'] == ModelBuilder.LAYER_OUTPUT:
            self.build_model[0]['shape'] = [None, self.state_size]
            self.build_model[-1]['length'] = self.action_size

        #Load each layer
        self.model_layers = []
        for layer_model in self.build_model:
            if layer_model['type'] == ModelBuilder.LAYER_INPUT: 
                if self.build_model.index(layer_model) == 0:
                    self.model_layers.append(placeholder(dtype=tf.float32, 
                        shape=layer_model['shape'], name='inputs_'))
                else:
                    raise IncoherentBuildModelError("Input Layer must be the first one.") 
            elif layer_model['type'] == ModelBuilder.LAYER_FULLY_CONNECTED:
                self.model_layers.append(layers.dense(inputs=self.model_layers[-1], 
                    units=layer_model['nodes'], 
                    activation=tf.nn.relu, name=layer_model['name']))
            elif layer_model['type'] == ModelBuilder.LAYER_OUTPUT:
                self.model_layers.append(layers.dense(inputs=self.model_layers[-1], 
                    units=self.action_size,activation=None))
            else:
                raise UnsupportedBuildModelLayerTypeError("Unsuported Layer Type " + layer_model['type'])

        #Setup output qsa layer and loss
        self.tf_qsa = placeholder(shape=[None, self.action_size], dtype=tf.float32)
        self.loss = tf.losses.mean_squared_error(self.tf_qsa, self.model_layers[-1])
        self.optimizer = train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        self.sess.run(global_variables_initializer())

        self.saver = train.Saver()
