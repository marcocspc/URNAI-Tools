'''
This file should contain all reward implementations that are generic enough to fit any agent/environment.
'''
from .abreward import RewardBase

class PureReward(RewardBase):
    def get_reward(self, obs, reward, done):
        return reward