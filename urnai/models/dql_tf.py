import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.compat.v1 import Session,ConfigProto,placeholder,layers,train,global_variables_initializer
import numpy as np
import random
import os
import pickle
from .base.abmodel import LearningModel
from agents.actions.base.abwrapper import ActionWrapper
from agents.states.abstate import StateBuilder

class DQLTF(LearningModel):
    def __init__(self, action_wrapper: ActionWrapper, state_builder: StateBuilder, save_path='urnai/models/saved/', file_name='temp_dqltf', learning_rate=0.001, gamma=0.90, decay_rate = 0.00001, name='DQN', nodes_layer1=10, nodes_layer2=10, nodes_layer3=10, nodes_layer4=10):
        super(DQLTF, self).__init__(action_wrapper, state_builder, gamma, learning_rate, save_path, file_name, name)

        if save_path is None:
            raise TypeError
        if file_name is None:
            raise TypeError

        self.full_save_path = "../" + self.save_path + os.path.sep + self.file_name 

        # EXPLORATION PARAMETERS FOR EPSILON GREEDY STRATEGY
        self.explore_start = 1.0
        self.explore_stop = 0.05
        self.decay_rate = decay_rate
        self.decay_step = 0

        # Number of Nodes of each Layer of our model
        self.nodes_layer1 = nodes_layer1
        self.nodes_layer2 = nodes_layer2
        self.nodes_layer3 = nodes_layer3
        self.nodes_layer4 = nodes_layer4

        self.pickle_obj = [self.decay_step, self.nodes_layer1, self.nodes_layer2, self.nodes_layer3, self.nodes_layer4]

        self.save_path = save_path
        self.file_name = file_name

        # Attempting to Load our serialized variables, as some of them will be used during the definition of our model
        # self.load_pickle()
        
        ops.reset_default_graph()

        tf.compat.v1.disable_eager_execution()

        # Initializing TensorFlow session
        self.sess = Session(config=ConfigProto(allow_soft_placement=True))

        self.inputs_ = placeholder(dtype=tf.float32, shape=[None, self.state_size], name='inputs_')
        self.actions_ = placeholder(dtype=tf.float32, shape=[None, self.action_size], name='actions_')
        

        # Defining the model's layers
        self.fc1 = layers.dense(inputs=self.inputs_,
                                   units=self.nodes_layer1,
                                   activation=tf.nn.relu,
                                   name='fc1')

        self.fc2 = layers.dense(inputs=self.fc1,
                                   units=self.nodes_layer2,
                                   activation=tf.nn.relu,
                                   name='fc2')

        self.fc3 = layers.dense(inputs=self.fc2,
                                   units=self.nodes_layer3,
                                   activation=tf.nn.relu,
                                   name='fc3')

        self.fc4 = layers.dense(inputs=self.fc3,
                                   units=self.nodes_layer4,
                                   activation=tf.nn.relu,
                                   name='fc4')                                                                      

        self.output_layer = layers.dense(inputs=self.fc4,
                                            units=self.action_size,
                                            activation=None)

        self.tf_qsa = placeholder(shape=[None, self.action_size], dtype=tf.float32)
        self.loss = tf.losses.mean_squared_error(self.tf_qsa, self.output_layer)
        self.optimizer = train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        self.sess.run(global_variables_initializer())

        self.saver = train.Saver()
        #self.load_extra(self.full_save_path)

    def learn(self, s, a, r, s_, done, is_last_step: bool):
        qsa_values = self.sess.run(self.output_layer, feed_dict={self.inputs_: s})

        current_q = 0

        if done:
            current_q = r
        else:
            current_q = r + self.gamma * self.__maxq(s_)

        qsa_values[0, a] = current_q

        self.sess.run(self.optimizer, feed_dict={self.inputs_: s, self.tf_qsa: qsa_values})

        qsa_values = self.sess.run(self.output_layer, feed_dict={self.inputs_: s})

    def __maxq(self, state):
        values = self.sess.run(self.output_layer, feed_dict={self.inputs_: state})

        index = np.argmax(values[0])
        mxq = values[0, index]

        return mxq

    def choose_action(self, state, excluded_actions=[]):
        self.decay_step += 1

        expl_expt_tradeoff = np.random.rand()

        explore_probability = self.explore_stop + (self.explore_start - self.explore_stop) * np.exp(-self.decay_rate * self.decay_step)

        if explore_probability > expl_expt_tradeoff:
            random_action = random.choice(self.actions)

            # Removing excluded actions
            while random_action in excluded_actions:
                random_action = random.choice(self.actions)
            action = random_action
        else:
            action = self.predict(state, excluded_actions)

        return action

    def predict(self, state, excluded_actions=[]):
        q_values = self.sess.run(self.output_layer, feed_dict={self.inputs_: state})
        action_idx = np.argmax(q_values)

        # Removing excluded actions
        # This is possibly badly optimized, eventually look back into this
        while action_idx in excluded_actions:
            q_values = np.delete(q_values, action_idx)
            action_idx = np.argmax(q_values)
        
        action = self.actions[int(action_idx)]
        return action

    def save_pickle(self, persist_path):
        self.pickle_obj = [self.decay_step, self.nodes_layer1, self.nodes_layer2, self.nodes_layer3, self.nodes_layer4]

        with open(self.get_full_persistance_pickle_path(persist_path), "wb") as pickle_out: 
            pickle.dump(self.pickle_obj, pickle_out)

    def save_extra(self, persist_path):
        self.saver.save(self.sess, self.get_full_persistance_tensorflow_path(persist_path))

    def load_extra(self, persist_path):
        self.decay_step = self.pickle_obj[0]
        self.nodes_layer1 = self.pickle_obj[1]
        self.nodes_layer2 = self.pickle_obj[2]

        tf_path = self.get_full_persistance_tensorflow_path(persist_path)
        exists = os.path.isfile(self.get_full_persistance_tensorflow_path(persist_path)+".meta")
        #If yes, load it
        if exists:
            self.saver.restore(self.sess, self.get_full_persistance_tensorflow_path(persist_path))
