from urnai.agents.rewards.abreward import RewardBuilder
from urnai.utils.constants import RTSGeneralization, Games 
import urnai.agents.actions.sc2 as sc2aux 
from pysc2.lib import units as sc2units
import numpy as np

class CollectablesGeneralizedRewardBuilder(RewardBuilder):

    def __init__(self, method=RTSGeneralization.STATE_MAP):
        self.previous_state = None
        self.method = method

    def get_game(self, obs):
        try:
            a = obs.feature_minimap
            return Games.SC2
        except AttributeError as ae:
            if "feature_minimap" in str(ae):
                return Games.DRTS

    def get_reward(self, obs, reward, done):
        reward = 0
        if self.previous_state != None: 
            game = self.get_game(obs)
            if game == Games.DRTS:
                try:
                    tmp = self.previous_state['units']
                except KeyError as ke:
                    if "units" in str(ke):
                        self.previous_state = obs
                reward = self.get_drts_reward(obs)
            else:
                try:
                    tmp = self.previous_state.feature_minimap
                except AttributeError as ae:
                    if "feature_minimap" in str(ae):
                        self.previous_state = obs
                reward = self.get_sc2_reward(obs)

        self.previous_state = obs
        return reward

    def get_drts_reward(self, obs):
        current = obs['collectables_map']
        prev = self.previous_state['collectables_map']
        curr = np.count_nonzero(current == 1)
        old = np.count_nonzero(prev == 1)
#        print(">>>>>>>>>>>> CURR: {}".format(curr))
#        print(">>>>>>>>>>>> OLD: {}".format(old))
        return (curr - old) * 1000

    def get_sc2_reward(self, obs):
        #layer 4 is units (1 friendly, 2 enemy, 16 mineral shards, 3 neutral 
        current = self.filter_non_mineral_shard_units(obs)
        prev = self.filter_non_mineral_shard_units(self.previous_state)
        return np.sum(current != prev) * 1000

    def filter_non_mineral_shard_units(self, obs):
        filtered_map = np.copy(obs.feature_minimap[4])
        for y in range(len(filtered_map)):
            for x in range(len(filtered_map[y])):
                element = filtered_map[y][x]
                if element != 16 or element != 0:
                    filtered_map[y][x] = 0

        return filtered_map


    def get_drts_player_units(self, obs, player):
        units = []
        for unit in obs["units"]:
            if unit.get_player() == obs["players"][player]:
                units.append(unit)

        return units

    def get_drts_player_specific_type_units(self, obs, player, unit_id):
        all_units = self.get_drts_player_units(obs, player)
        specific_units = []

        for unit in all_units:
            if int(unit.type) == unit_id: 
                specific_units.append(unit)

        return specific_units

    def get_drts_number_of_specific_units(self, obs, player, unit_id):
        return len(self.get_drts_player_specific_type_units(obs, player, unit_id))

    def get_sc2_number_of_zerglings(self, obs):
        units = sc2aux.get_units_by_type(obs, sc2units.Zerg.Zergling)
        return len(units)

    def get_sc2_number_of_roaches(self, obs):
        units = sc2aux.get_units_by_type(obs, sc2units.Zerg.Roach)
        return len(units)

    def get_sc2_number_of_marines(self, obs):
        units = sc2aux.get_units_by_type(obs, sc2units.Terran.Marine)
        return len(units)

    def get_sc2_number_of_barracks(self, obs):
        units = sc2aux.get_units_by_type(obs, sc2units.Terran.Barracks)
        return len(units)

    def get_sc2_number_of_supply_depot(self, obs):
        units = sc2aux.get_units_by_type(obs, sc2units.Terran.SupplyDepot)
        return len(units)


