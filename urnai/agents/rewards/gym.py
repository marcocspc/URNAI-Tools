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
        self.sumReward += (reward - 16/(np.count_nonzero(obs==0)+1))
        if(self.sparce):
            if(done):
                totalReward = self.sumReward
                self.sumReward = 0
                return totalReward
            else:
                return 0
        else:
            return (reward - 16/(np.count_nonzero(obs==0)+1))
