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
    def __init__(self, action_wrapper: ActionWrapper, state_builder: StateBuilder, save_path='urnai/models/saved/', file_name='temp', learning_rate=0.001, gamma=0.90, name='DQN', nodes_layer1=10, nodes_layer2=10):
        super(DQLTF, self).__init__(action_wrapper, state_builder, gamma, learning_rate, save_path, file_name, name)

        if save_path is None:
            raise TypeError
        if file_name is None:
            raise TypeError

        # EXPLORATION PARAMETERS FOR EPSILON GREEDY STRATEGY
        self.explore_start = 1.0
        self.explore_stop = 0.05
        self.decay_rate = 0.00001
        self.decay_step = 0

        # Number of Nodes of each Layer of our model
        self.nodes_layer1 = nodes_layer1
        self.nodes_layer2 = nodes_layer2

        self.pickle_obj = [self.decay_step, self.nodes_layer1, self.nodes_layer2]

        self.save_path = save_path
        self.file_name = file_name

        # Attempting to Load our serialized variables, as some of them will be used during the definition of our model
        self.load_pickle()
        
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

        self.output_layer = layers.dense(inputs=self.fc2,
                                            units=self.action_size,
                                            activation=None)

        self.tf_qsa = placeholder(shape=[None, self.action_size], dtype=tf.float32)
        self.loss = tf.losses.mean_squared_error(self.tf_qsa, self.output_layer)
        self.optimizer = train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        self.sess.run(global_variables_initializer())

        self.saver = train.Saver()
        self.load()

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

    def save(self, save_path, file_name):
        print("\n> Saving the model!\n")
        self.saver.save(self.sess, self.save_path+self.file_name+"/"+self.file_name)

        # Dumping (serializing) decay_step into a pickle file
        pickle_out = open(self.save_path+self.file_name+"/"+self.file_name+"_model.pickle", "wb")
        self.pickle_obj = [self.decay_step, self.nodes_layer1, self.nodes_layer2]
        pickle.dump(self.pickle_obj, pickle_out)
        pickle_out.close()

    def load(self):
        exists = os.path.isfile(self.save_path + self.file_name + "/" + self.file_name + '.meta')
        if exists:
            print("\n> Loading saved model!\n")
            self.saver.restore(self.sess, self.save_path + self.file_name + "/" + self.file_name)

    def load_pickle(self):
    # Loading (deserializing) a few parameters from a pickle file 
        exists_pickle = os.path.isfile(self.save_path + self.file_name + "/" + self.file_name + '_model.pickle')
        if exists_pickle:
            pickle_in  = open(self.save_path + self.file_name + "/" + self.file_name + "_model.pickle", "rb")
            self.pickle_obj = pickle.load(pickle_in)
            self.decay_step = self.pickle_obj[0]
            self.nodes_layer1 = self.pickle_obj[1]
            self.nodes_layer2 = self.pickle_obj[2]
