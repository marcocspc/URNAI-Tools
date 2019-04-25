import tensorflow as tf
import numpy as np
import random
import os
from .base.abmodel import LearningModel
from agents.actions.base.abwrapper import ActionWrapper
from agents.states.abstate import State

# EXPLORATION PARAMETERS FOR EPSILON GREEDY STRATEGY
explore_start = 1.0
explore_stop = 0.01
decay_rate = 0.0001

class DQNWorking(LearningModel):
    def __init__(self, action_wrapper: ActionWrapper, state_builder: State, save_path, learning_rate=0.0002, gamma=0.95, name='DQN'):
        super(DQNWorking, self).__init__(action_wrapper, state_builder, save_path, name)

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.decay_step = 0

        tf.reset_default_graph()
        
        # Initializing our TensorFlow session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

        self.inputs_ = tf.placeholder(dtype=tf.float32, shape=[None, self.state_size], name='inputs_')
        self.actions_ = tf.placeholder(dtype=tf.float32, shape=[None, self.action_size], name='actions_')

        # Defining the layers of our neural network
        self.fc1 = tf.layers.dense(inputs=self.inputs_,
                                    units=50,
                                    activation=tf.nn.relu,
                                    name='fc1')

        self.fc2 = tf.layers.dense(inputs=self.fc1,
                                    units=50,
                                    activation=tf.nn.relu,
                                    name='fc2')

        self.output_layer = tf.layers.dense(inputs=self.fc2,
                                        units=self.action_size,
                                        activation=None)

        self.tf_qsa = tf.placeholder(shape=[None, self.action_size], dtype=tf.float32)
        self.loss = tf.losses.mean_squared_error(self.tf_qsa, self.output_layer)
        self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)

        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()
        self.load()


    def learn(self, s, a, r, s_, done):
        qsa_values = self.sess.run(self.output_layer, feed_dict={self.inputs_: s})

        current_q = 0

        if done:
            current_q = r
        else:
            current_q = r + self.gamma * self.maxq(s_)
        
        qsa_values[0, a] = current_q

        self.sess.run(self.optimizer, feed_dict={self.inputs_: s, self.tf_qsa: qsa_values})

        qsa_values = self.sess.run(self.output_layer, feed_dict={self.inputs_: s})


    def maxq(self, state):
        values = self.sess.run(self.output_layer, feed_dict={self.inputs_: state})

        index = np.argmax(values[0])
        mxq = values[0, index]

        return mxq


    def choose_action(self, state, excluded_actions=[]):
        self.decay_step += 1

        expl_expt_tradeoff = np.random.rand()

        explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * self.decay_step)

        # TODO: Exclude actions
        if explore_probability > expl_expt_tradeoff:
            action = random.choice(self.actions)
        else:
            action = self.predict(state)

        return action


    def predict(self, state):
        q_values = self.sess.run(self.output_layer, feed_dict={self.inputs_: state})
        action_idx = np.argmax(q_values)
        action = self.actions[int(action_idx)]
        return action


    def save(self):
        print()
        print("> Saving the model!")
        print()
        self.saver.save(self.sess, self.save_path)

    def load(self):
        exists = os.path.isfile(self.save_path + '.meta')
        if exists:
            print()
            print("> Loading saved model!")
            print()
            self.saver.restore(self.sess, self.save_path)
