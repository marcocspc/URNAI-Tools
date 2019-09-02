'''
This file is a repository with reward classes for all gym games we've solved.
'''
from .abreward import RewardBuilder
import numpy as np

class FrozenlakeReward(RewardBuilder):
    def get_reward(self, obs, reward, done):
        if reward == 1:
            return 1000
        elif reward == 0:
            return 1
        else:
            return -1000

class Game2048Reward(RewardBuilder):
    def __init__(self, sparce=False):
        self.sumReward = 0
        self.sparce = sparce

    def get_reward(self, obs, reward, done):
        self.sumReward += self.div_reward(obs, reward, done)
        if(self.sparce):
            if(done):
                totalReward = self.sumReward
                self.sumReward = 0
                return totalReward
            else:
                return 0
        else:
            # TO DO: Return negative if game lost
            return self.div_reward(obs, reward, done)

    # divided reward is calculated based on the standard reward from the game minus the amount of spaces (16 for a 4x4 grid) divided by the amount of empty spaces plus 1
    # div_reward = reward - (spaces/(empty spaces + 1))
    def div_reward(self, obs, reward, done):
        return (reward - obs.size/(np.count_nonzero(obs==0)+1))
    
    #standard reward provided by the gym environment    
    def std_reward(self, obs, reward, done):
        return reward
