import pandas as pd
import numpy as np

import os.path

class QLearningTable:
    def __init__(self, actions, save_path, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=range(len(self.actions)), dtype=np.float64)     # Table columns are named by the indices of actions
        self.disallowed_actions = {}    # Dictionary that maps observation -> invalid actions
        self.save_path = save_path


    def choose_action(self, state, excluded_actions=[]):

        state_str = str(state)

        self.check_state_exist(state_str)

        self.disallowed_actions[state_str] = excluded_actions

        # Actions for the current state
        state_actions = self.q_table.ix[state_str, :]

        for excluded_action in excluded_actions:
            del state_actions[np.argmax(excluded_action)]

        if np.random.uniform() < self.epsilon:
            ## Exploitation: choosing the best action

            # Some actions have the same value
            state_actions = state_actions.reindex(np.random.permutation(state_actions.index))
            action_idx = state_actions.idxmax()
        else:
            # Exploration: Choosing a random action
            action_idx = np.random.choice(state_actions.index)

        return self.actions[action_idx]

    
    def learn(self, s, a, r, s_, done):
        state = str(s)
        state_ = str(s_)
        # The agent frequently lands on the same state because its representation
        # is very simple. This tends to push less valuable actions up towards to
        # the most valuable action. We're aborting the learning when both states
        # are identical.
        if state == state_:
            return

        self.check_state_exist(state)
        self.check_state_exist(state_)

        a_idx = np.argmax(a)

        q_predict = self.q_table.ix[state, a_idx]

        s_rewards = self.q_table.ix[state_, :]

        # Since invalid actions never get chosen, they always have 0 value for
        # that state. If all the other actions have negative values, invalid
        # actions will be chosen, even though they can't be used. So we're
        # filtering these actions out of learning, from the future state's reward.
        if state_ in self.disallowed_actions:
            for excluded_action in self.disallowed_actions[state_]:
                excluded_action_idx = np.argmax(excluded_action)
                del s_rewards[excluded_action_idx]

        if state_ != 'terminal':
            q_target = r + self.gamma * s_rewards.max()
        else:
            q_target = r
        
        # Updating our reward
        self.q_table.ix[state, a] += self.lr * (q_target - q_predict)


    def check_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))


    def load(self):
        if os.path.isfile(self.save_path + '.gz'):
            self.q_table = pd.read_pickle(self.save_path + '.gz', compression='gzip')


    def save(self):
        self.q_table.to_pickle(self.save_path + '.gz', 'gzip')

