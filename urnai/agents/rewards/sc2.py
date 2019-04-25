'''
This file is a repository with reward classes for all StarCraft 2 games/minigames we've solved.
'''
from .abreward import RewardBase

class SparseReward(RewardBase):
    def get_reward(self, obs, reward, done):
        '''
        Always returns 0, unless the game has ended.
        '''
        if not done:
            return 0
        return reward


class KilledUnitsReward(RewardBase):
    def __init__(self):
        # Properties keep track of the change of values used in our reward system
        self._previous_killed_unit_score = 0
        self._previous_killed_building_score = 0

        self._kill_unit_reward = 0.2
        self._kill_building_reward = 0.5

    # When the episode is over, the values we use to compute our reward should be reset.
    def reset(self):
        self._previous_killed_unit_score = 0
        self._previous_killed_building_score = 0

    def get_reward(self, obs, reward, done):
        # Getting values from PySC2's cumulative score system
        current_killed_unit_score = obs.score_cumulative[5]
        current_killed_building_score = obs.score_cumulative[6]

        reward = 0

        if current_killed_unit_score > self._previous_killed_unit_score:
            reward += self._kill_unit_reward

        if current_killed_building_score > self._previous_killed_building_score:
            reward += self._kill_building_reward

        # Saving the previous values for killed units and killed buildings.
        self._previous_killed_unit_score = current_killed_unit_score
        self._previous_killed_building_score = current_killed_building_score

        if done:
            self.reset()

        return reward
