from urnai.agents.actions import sc2 as scaux
from .defeatenemies import DefeatEnemiesDeepRTSActionWrapper, DefeatEnemiesStarcraftIIActionWrapper  
from pysc2.lib import actions, features, units
from statistics import mean
from pysc2.env import sc2_env
import random

class BuildUnitsDeepRTSActionWrapper(DefeatEnemiesDeepRTSActionWrapper):
    def __init__(self):
        super().__init__()
        self.collect_gold = 17
        self.build_farm = 18
        self.build_barrack = 19
        self.build_footman = 20

        self.actions = [self.previousunit, self.nextunit, self.moveleft, self.moveright, self.moveup, self.movedown,
                self.moveupleft, self.moveupright, self.movedownleft, self.movedownright, self.attack, self.harvest,
                self.build0, self.build1, self.build2, self.noaction, self.run, self.collect_gold, self.build_farm, 
                self.build_barrack, self.build_footman] 

        self.excluded_actions = [self.previousunit, self.nextunit, self.moveleft, self.moveright, self.moveup, self.movedown,
                self.moveupleft, self.moveupright, self.movedownleft, self.movedownright, self.attack, self.harvest,
                self.build0, self.build1, self.build2, self.noaction, self.run] 

        self.final_actions = list(set(self.actions) - set(self.excluded_actions))

    def solve_action(self, action_idx, obs):
        if action_idx == self.run:
            self.run_(obs)
        elif action_idx == self.attack:
            self.attack_(obs)

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

    SUPPLY_DEPOT_X = 42
    SUPPLY_DEPOT_Y = 42
    BARRACK_X = 39
    BARRACK_Y = 36

    def __init__(self):
        super().__init__()

        self.collect_minerals = 7
        self.build_supply_depot = 8
        self.build_barrack = 9
        self.build_marine = 10
        self.actions = [self.collect_minerals, self.build_supply_depot, self.build_barrack, self.build_marine, self.stop]

    def solve_action(self, action_idx, obs):
        if action_idx == self.collect_minerals:
            self.collect(obs)
        elif action_idx == self.build_supply_depot:
            self.build_supply_depot_(obs)
        elif action_idx == self.build_barrack:
            self.build_barrack_(obs)
        elif action_idx == self.build_marine:
            self.build_marine_(obs)
        elif action_idx == self.stop:
            self.pending_actions.clear()

    def collect(self, obs):
        #get SCV list
        scvs = scaux.get_my_units_by_type(obs, units.Terran.SCV)
        #get mineral list
        mineral_fields = scaux.get_neutral_units_by_type(obs, units.Neutral.MineralField)
        #split SCVs into sets of numberSCVs/numberOfMinerals
        n = int(len(scvs)/len(mineral_fields))
        scvs_sets = [scvs[i * n:(i + 1) * n] for i in range((len(scvs) + n - 1) // n )]
        #make every set of SCVs collect one mineral 
        for i in range(len(mineral_fields)):
            mineral = mineral_fields[i]
            scvset = scvs_sets[i]
            for scv in scvset:
                self.pending_actions.append(actions.RAW_FUNCTIONS.Harvest_Gather_unit("queued", scv.tag, mineral.tag))

    def select_random_scv(self, obs):
        #get SCV list
        scvs = scaux.get_my_units_by_type(obs, units.Terran.SCV)
        length = len(scvs)
        scv = scvs[random.randint(0, length - 1)] 
        return scv

    def build_supply_depot_(self, obs):
        #randomly select scv
        scv = self.select_random_scv(obs)
        #get coordinates
        x, y = BuildUnitsStarcraftIIActionWrapper.SUPPLY_DEPOT_X, BuildUnitsStarcraftIIActionWrapper.SUPPLY_DEPOT_Y
        #append action to build supply depot
        self.pending_actions.append(actions.RAW_FUNCTIONS.Build_SupplyDepot_pt("now", scv.tag, [x, y]))

    def build_barrack_(self, obs):
        #randomly select scv
        scv = self.select_random_scv(obs)
        #get coordinates
        x, y = BuildUnitsStarcraftIIActionWrapper.BARRACK_X, BuildUnitsStarcraftIIActionWrapper.BARRACK_Y 
        #append action to build supply depot
        self.pending_actions.append(actions.RAW_FUNCTIONS.Build_Barracks_pt("now", scv.tag, [x, y]))

    def build_marine_(self, obs):
        barracks = scaux.get_my_units_by_type(obs, units.Terran.Barracks)
        for barrack in barracks:
            self.pending_actions.append(actions.RAW_FUNCTIONS.Train_Marine_quick("now", barrack.tag))
