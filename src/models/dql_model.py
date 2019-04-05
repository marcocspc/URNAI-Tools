import tensorflow as tf
import numpy as np
import random
import os.path
from .base.abmodel import LearningModel
from agents.actions.base.abwrapper import ActionWrapper

# TRAINING HYPERPARAMETERS
max_steps = 100
batch_size = 64

# EXPLORATION PARAMETERS FOR EPSILON GREEDY STRATEGY
explore_start = 1.0
explore_stop = 0.01
decay_rate = 0.0001

# MEMORY HYPERPARAMETERS
# The number of experience tuples stored in memory when initializing it
pretrain_length = batch_size
memory_size = 1000000   # Maximum number of experience tuples the memory can keep

# If training is set to False, we'll just see the trained agent playing it's optimal policy
training = True


class DQNetwork(LearningModel):
    def __init__(self, agent, save_path, learning_rate=0.0002, gamma=0.95, name='DQNetwork'):
        super(DQNetwork, self).__init__(agent, save_path, name)

        self.learning_rate = learning_rate
        self.discount_rate = gamma
        self.decay_step = 0

        tf.reset_default_graph()
        self.sess = tf.Session()

        with tf.variable_scope(name):
            self.inputs_ = tf.placeholder(tf.float32, [None, *self.state_size], name='inputs')
            self.actions_ = tf.placeholder(tf.float32, [None, self.action_size], name='actions_')

            # targetQ = R(s, a) + Qhat(s', a')
            self.target_Q = tf.placeholder(tf.float32, [None], name='target')

            self.fc1 = tf.layers.dense(inputs=self.inputs_,
                                       units=50,
                                       activation=tf.nn.elu,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                       name='fc1')

            self.fc2 = tf.layers.dense(inputs=self.fc1,
                                       units=50,
                                       activation=tf.nn.elu,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                       name='fc2')

            self.fc3 = tf.layers.dense(inputs=self.fc2,
                                       units=50,
                                       activation=tf.nn.elu,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                       name='fc3')

            self.output = tf.layers.dense(inputs=self.fc3,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          units=self.action_size,
                                          activation=None)

            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=1)

            self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))

            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

        # Setting up tensorboard writer
        self.writer = tf.summary.FileWriter("/tensorboard/dqn/1")
        tf.summary.scalar('Loss', self.loss)
        self.write_op = tf.summary.merge_all()

        # Initializing our Variables
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.load()


    def learn(self, s, a, r, s_, done):
        if str(s) == str(s_):
            return

        if not done:
            Qs_next_state = self.sess.run(self.output,
                                          feed_dict={self.inputs_: s_.reshape((1, *s_.shape))})

            q_target = r + self.discount_rate * Qs_next_state.max()
        else:
            q_target = r    # The next state is terminal

        # TODO: Write this sess.run to loss
        self.sess.run(self.write_op,
                      feed_dict={
                          self.inputs_: s.reshape((1, *s.shape)),
                          self.target_Q: np.array([q_target]),
                          self.actions_: np.expand_dims(a, axis=0)})

        summary = self.sess.run(self.write_op,
                                feed_dict={
                                    self.inputs_: s.reshape((1, *s.shape)),
                                    self.target_Q: np.array([q_target]),
                                    self.actions_: np.expand_dims(a, axis=0)})

        #self.writer.add_summary(summary, episode_num)
        # self.writer.flush()


    # EPSILON GREEDY STRATEGY
    def choose_action(self, state, excluded_actions=[], is_playing=False):
        self.decay_step += 1

        expl_expt_tradeoff = np.random.rand()

        explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * self.decay_step)

        # TODO: Exclude actions
        if explore_probability > expl_expt_tradeoff and not is_playing:
            action = random.choice(self.actions)
        else:
            q_values = self.sess.run(self.output, feed_dict={self.inputs_: state.reshape((1, *state.shape))})

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
