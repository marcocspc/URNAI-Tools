'''
This file is a repository with reward classes for all StarCraft 2 games/minigames we've solved.
'''
from .abreward import RewardBuilder

class SparseReward(RewardBuilder):
    def get_reward(self, obs, reward, done):
        '''
        Always returns 0, unless the game has ended.
        '''
        if not done:
            return 0
        return reward

class GeneralReward(RewardBuilder):
    def __init__(self):
        self.reward = 0

        self.last_own_worker_count = 0
        self.last_own_army_count = 0
        self.last_structures_score = 0
        self.last_killed_units_score = 0
        self.last_killed_structures_score = 0

    def get_reward(self, obs, reward, done):
        currentscore = -1
        currentscore += (obs.player.food_army - self.last_own_army_count)*50
        currentscore += (obs.player.food_workers - self.last_own_worker_count)*25
        currentscore += obs.score_cumulative.total_value_structures - self.last_structures_score
        currentscore += (obs.score_cumulative.killed_value_units - self.last_killed_units_score)
        currentscore += (obs.score_cumulative.killed_value_structures - self.last_killed_structures_score)*2

        self.last_own_army_count = obs.player.food_army
        self.last_own_worker_count = obs.player.food_workers
        self.last_killed_units_score = obs.score_cumulative.killed_value_units
        self.last_killed_structures_score = obs.score_cumulative.killed_value_structures
        self.last_structures_score = obs.score_cumulative.total_value_structures

        self.reward = currentscore
        return self.reward

class KilledUnitsReward(RewardBuilder):
    def __init__(self):
        # Properties keep track of the change of values used in our reward system
        self._previous_killed_unit_score = 0
        self._previous_killed_building_score = 0

        self._kill_unit_reward = 1
        self._kill_building_reward = 2

    # When the episode is over, the values we use to compute our reward should be reset.
    def reset(self):
        self._previous_killed_unit_score = 0
        self._previous_killed_building_score = 0

    def get_reward(self, obs, reward, done):
        # Getting values from PySC2's cumulative score system
        current_killed_unit_score = obs.score_cumulative[5]
        current_killed_building_score = obs.score_cumulative[6]

        new_reward = 0

        if current_killed_unit_score > self._previous_killed_unit_score:
            new_reward += self._kill_unit_reward

        if current_killed_building_score > self._previous_killed_building_score:
            new_reward += self._kill_building_reward

        # Saving the previous values for killed units and killed buildings.
        self._previous_killed_unit_score = current_killed_unit_score
        self._previous_killed_building_score = current_killed_building_score

        if done:
            self.reset()

        return new_reward
