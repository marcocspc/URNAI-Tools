from urnai.agents.actions import sc2 as scaux
from .findanddefeat import FindAndDefeatDeepRTSActionWrapper, FindAndDefeatStarcraftIIActionWrapper 
from pysc2.lib import actions, features, units
from statistics import mean
from pysc2.env import sc2_env

class DefeatEnemiesDeepRTSActionWrapper(FindAndDefeatDeepRTSActionWrapper):
    def __init__(self):
        super().__init__()
        self.run = 17

        self.final_actions = [self.attack, self.run, self.cancel]
        self.action_indices = range(len(self.final_actions))



    def solve_action(self, action_idx, obs):
        if action_idx != self.noaction:
            i = action_idx 
            if self.final_actions[i] == self.run:
                self.run_(obs)
            else:
                super().solve_action(action_idx, obs)

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
        self.enqueue_action_for_player_units(obs, self.run)

        #its not this simple
        #p_army_x, p_army_y = self.get_army_mean(0, obs)
        #e_army_x, e_army_y = self.get_army_mean(1, obs)

        #if p_army_x - e_army_x < 0:
        #    self.enqueue_action_for_player_units(obs, self.moveleft)
        #else:
        #    self.enqueue_action_for_player_units(obs, self.moveright)

        #if p_army_y - e_army_y < 0:
        #    self.enqueue_action_for_player_units(obs, self.moveup)
        #else:
        #    self.enqueue_action_for_player_units(obs, self.movedown)


class DefeatEnemiesStarcraftIIActionWrapper(FindAndDefeatStarcraftIIActionWrapper):

    MAP_X_CORNER_LEFT = 21
    MAP_X_CORNER_RIGHT = 44
    MAP_Y_CORNER_UP = 27
    MAP_Y_CORNER_DOWN = 50

    def __init__(self):
        super().__init__()

        self.maximum_attack_range = 999999 
        self.ver_threshold = 3
        self.hor_threshold = 3

        self.run = 5
        self.actions = [self.attack, self.run, self.stop]
        self.action_indices = range(len(self.actions))

    def solve_action(self, action_idx, obs):
        if action_idx != self.noaction:
            action = self.actions[action_idx]
            if action == self.attack:
                self.attack_(obs)
            elif action == self.run:
                self.run_(obs)
            elif action == self.stop:
                self.pending_actions.clear()
        
    def run_(self, obs):
        #TODO
        #get player x and y avg
        p_army_x, p_army_y = self.get_race_unit_avg(obs, sc2_env.Race.terran)
        #get enemy x and y avg
        e_army_x, e_army_y = self.get_race_unit_avg(obs, sc2_env.Race.zerg)

        new_x = 0
        new_y = 0

        if p_army_x - e_army_x < 0:
            new_x = DefeatEnemiesStarcraftIIActionWrapper.MAP_X_CORNER_LEFT
        else:
            new_x = DefeatEnemiesStarcraftIIActionWrapper.MAP_X_CORNER_RIGHT

        if p_army_y - e_army_y < 0:
            new_y = DefeatEnemiesStarcraftIIActionWrapper.MAP_Y_CORNER_UP
        else:
            new_y = DefeatEnemiesStarcraftIIActionWrapper.MAP_Y_CORNER_DOWN

        army = scaux.select_army(obs, sc2_env.Race.terran)
        for unit in army:
            self.pending_actions.append(actions.RAW_FUNCTIONS.Move_pt("now", unit.tag, [new_x, new_y]))
