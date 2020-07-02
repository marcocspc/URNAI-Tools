from urnai.agents.actions import sc2 as scaux
from .findanddefeat import FindAndDefeatDeepRTSActionWrapper, FindAndDefeatStarcraftIIActionWrapper 
from pysc2.lib import actions, features, units
from statistics import mean
from pysc2.env import sc2_env

class DefeatEnemiesDeepRTSActionWrapper(FindAndDefeatDeepRTSActionWrapper):
    pass

class DefeatEnemiesStarcraftIIActionWrapper(FindAndDefeatStarcraftIIActionWrapper):
    def __init__(self):
        super().__init__()

        self.maximum_attack_range = 999999 
        self.ver_threshold = 3
        self.hor_threshold = 3

        self.run = 5
        self.actions = [self.attack, self.run]

    def solve_action(self, action_idx, obs):
        if action_idx == self.attack:
            self.attack_(obs)
        if action_idx == self.run:
            self.run_(obs)
        
    def run_(self, obs):
        #TODO
        #get player x and y avg
        avg_p_x, avg_p_y = self.get_race_unit_avg(obs, sc2_env.Race.terran)
        #get enemy x and y avg
        avg_e_x, avg_e_y = self.get_race_unit_avg(obs, sc2_env.Race.zerg)

        final_x = 0
        final_y = 0
        if avg_p_x - avg_e_x < 0:
            final_x = avg_p_x - self.hor_threshold
        else:
            final_x = avg_p_x + self.hor_threshold

        if avg_p_y - avg_e_y < 0:
            final_y = avg_p_y - self.ver_threshold
        else:
            final_y = avg_p_y + self.ver_threshold

        army = scaux.select_army(obs, sc2_env.Race.terran)
        for unit in army:
            self.pending_actions.append(actions.RAW_FUNCTIONS.Move_pt("now", unit.tag, [final_x, final_y]))
