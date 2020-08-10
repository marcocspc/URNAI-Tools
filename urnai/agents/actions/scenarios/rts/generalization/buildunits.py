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

        self.collect_minerals = 6
        self.build_supply_depot = 7
        self.build_barrack = 8
        self.build_marine = 9
        self.actions = [self.collect_minerals, self.build_supply_depot, self.build_barrack, self.build_marine]

    def solve_action(self, action_idx, obs):
        if action_idx == self.collect_minerals:
            self.collect(obs)
        elif action_idx == self.build_supply_depot:
            self.build_supply_depot_(obs)
        elif action_idx == self.build_barrack:
            self.build_barrack_(obs)
        elif action_idx == self.build_marine:
            self.build_marine_(obs)
