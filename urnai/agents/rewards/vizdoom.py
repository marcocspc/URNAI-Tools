from .abreward import RewardBuilder
import numpy

class VizDoomDefaultReward(RewardBuilder):

    def get_reward(self, obs, reward, done):
        return reward

class VizDoomHealthGatheringReward(RewardBuilder):

    def __init__(self):
        self.prev_health = 0


    KILLCOUNT = 0
    ITEMCOUNT = 1
    SECRETCOUNT = 2
    DEATHCOUNT = 3
    HITCOUNT = 4
    HITS_TAKEN = 5
    DAMAGECOUNT = 6
    DAMAGE_TAKEN = 7
    HEALTH = 8
    ARMOR = 9
    DEAD = 10
    ATTACK_READY = 11
    SELECTED_WEAPON = 12
    SELECTED_WEAPON_AMMO = 13
    POSITION_X = 14
    POSITION_Y = 15
    POSITION_Z = 16 
    GENERAL_REWARD = 17
    
    METHOD_CUMULATIVE = "cumulative"
    METHOD_DIFFERENCE = "difference"
    METHOD_POSITIVE_ONLY = "positive_only"

    def __init__(self, method):
        self.method = method 

    def get_reward(self, obs, reward, done):
        r = 0
        
        if self.method == VizDoomHealthGatheringReward.METHOD_CUMULATIVE:
            r += obs.game_variables[VizDoomHealthGatheringReward.HEALTH]
        elif self.method == VizDoomHealthGatheringReward.METHOD_DIFFERENCE:
            r += obs.game_variables[VizDoomHealthGatheringReward.HEALTH] - self.prev_health 
            self.prev_health = obs.game_variables[VizDoomHealthGatheringReward.HEALTH]
        elif self.method == VizDoomHealthGatheringReward.METHOD_POSITIVE_ONLY:
            r += obs.game_variables[VizDoomHealthGatheringReward.HEALTH] - self.prev_health 
            self.prev_health = obs.game_variables[VizDoomHealthGatheringReward.HEALTH]

            if r < 0: r = 0

        return r


class VizDoom2CustomReward(RewardBuilder):

    KILLCOUNT = 0
    ITEMCOUNT = 1
    SECRETCOUNT = 2
    DEATHCOUNT = 3
    HITCOUNT = 4
    HITS_TAKEN = 5
    DAMAGECOUNT = 6
    DAMAGE_TAKEN = 7
    HEALTH = 8
    ARMOR = 9
    DEAD = 10
    ATTACK_READY = 11
    SELECTED_WEAPON = 12
    SELECTED_WEAPON_AMMO = 13
    POSITION_X = 14
    POSITION_Y = 15
    POSITION_Z = 16 
    GENERAL_REWARD = 17

    def get_reward(self, obs, reward, done):
        r = 0
        
        r += 10 * obs.game_variables[VizDoom2CustomReward.HITCOUNT]        
        r += 50 * obs.game_variables[VizDoom2CustomReward.KILLCOUNT]
        r += -10 * obs.game_variables[VizDoom2CustomReward.DAMAGE_TAKEN]
        r += -100 * obs.game_variables[VizDoom2CustomReward.DEAD]
        r += 10 * obs.game_variables[VizDoom2CustomReward.SELECTED_WEAPON_AMMO] 

        return r

