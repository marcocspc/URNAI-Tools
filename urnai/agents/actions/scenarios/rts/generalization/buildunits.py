from urnai.agents.actions import sc2 as scaux
from .defeatenemies import DefeatEnemiesDeepRTSActionWrapper, DefeatEnemiesStarcraftIIActionWrapper  
from pysc2.lib import actions, features, units
from statistics import mean
from pysc2.env import sc2_env

class BuildUnitsDeepRTSActionWrapper(DefeatEnemiesDeepRTSActionWrapper):
    def __init__(self):
        super().__init__()
        self.run = 16

        self.actions = [self.previousunit, self.nextunit, self.moveleft, self.moveright, self.moveup, self.movedown,
                self.moveupleft, self.moveupright, self.movedownleft, self.movedownright, self.attack, self.harvest,
                self.build0, self.build1, self.build2, self.noaction, self.run] 

        self.excluded_actions = [self.previousunit, self.nextunit, self.moveleft, self.moveright, self.moveup, self.movedown,
                self.moveupleft, self.moveupright, self.movedownleft, self.movedownright, self.harvest,
                self.build0, self.build1, self.build2, self.noaction] 

        self.final_actions = list(set(self.actions) - set(self.excluded_actions))

    def solve_action(self, action_idx, obs):
        if action_idx == self.run:
            self.run_(obs)
        elif action_idx == self.attack:
            self.attack_(obs)

    def get_army_mean(self, player, obs):
        xs = []
        ys = []

        for unit in self.get_player_units(obs["players"][player], obs):
            try:
                xs.append(unit.tile.x)
                ys.append(unit.tile.y)
            except AttributeError as ae:
                if not "'NoneType' object has no attribute 'x'" in str(ae):
                    raise

        army_x = int(mean(xs))
        army_y = int(mean(ys))
        return army_x, army_y

    def run_(self, obs):
        #its not this simple
        p_army_x, p_army_y = self.get_army_mean(0, obs)
        e_army_x, e_army_y = self.get_army_mean(1, obs)

        if p_army_x - e_army_x < 0:
            self.enqueue_action_for_player_units(obs, self.moveleft)
        else:
            self.enqueue_action_for_player_units(obs, self.moveright)

        if p_army_y - e_army_y < 0:
            self.enqueue_action_for_player_units(obs, self.moveup)
        else:
            self.enqueue_action_for_player_units(obs, self.movedown)


class BuildUnitsStarcraftIIActionWrapper(DefeatEnemiesStarcraftIIActionWrapper):
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
