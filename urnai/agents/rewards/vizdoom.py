from .abreward import RewardBuilder
import numpy

class VizDoomDefaultReward(RewardBuilder):

    def get_reward(self, obs, reward, done):
        return reward

class VizDoomHealthGatheringReward(RewardBuilder):

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
        
        r += -10 * obs.game_variables[VizDoomHealthGatheringReward.DEAD]
        r += 15 * obs.game_variables[VizDoomHealthGatheringReward.ITEMCOUNT] 
        r += 10 * obs.game_variables[VizDoomHealthGatheringReward.HEALTH] 
        r += -10 * reward

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

