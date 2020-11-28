from .defeatenemies import DefeatEnemiesGeneralizedRewardBuilder 
from utils.constants import RTSGeneralization 

class BuildUnitsGeneralizedRewardBuilder(DefeatEnemiesGeneralizedRewardBuilder):

    LAST_CHOSEN_ACTION = -1
    ACTION_DO_NOTHING = 7
    ACTION_BUILD_SUPPLY_DEPOT = 8
    ACTION_BUILD_BARRACK = 9
    ACTION_BUILD_MARINE = 10

    def get_drts_reward(self, obs):
        player = 0
        footman = 5
        farm = 6
        barracks = 4
        build_farm = RTSGeneralization.ACTION_DRTS_BUILD_FARM
        build_barrack = RTSGeneralization.ACTION_DRTS_BUILD_BARRACK
        build_footman = RTSGeneralization.ACTION_DRTS_BUILD_FOOTMAN

        #current = self.get_drts_number_of_specific_units(obs, player, farm) 
        #prev = self.get_drts_number_of_specific_units(self.previous_state, player, farm) 

        #rwdA = (current - prev)

        #current = self.get_drts_number_of_specific_units(obs, player, barracks) 
        #prev = self.get_drts_number_of_specific_units(self.previous_state, player, barracks) 

        #rwdB = (current - prev) * 10

        rwdA = 0
        chosen_action = BuildUnitsGeneralizedRewardBuilder.LAST_CHOSEN_ACTION 
        #print(chosen_action)
        if chosen_action > -1:
            farm_number = self.get_drts_number_of_specific_units(obs, player, farm) 
            barracks_amount = self.get_drts_number_of_specific_units(obs, player, barracks)
            gold_amount = obs['players'][0].gold 
            if chosen_action == build_farm: 
                if farm_number > 7 or gold_amount < 500:
                    rwdA = -1
            elif chosen_action == build_barrack:
                if farm_number <= 0 or gold_amount < 700:
                    rwdA = -1
            elif chosen_action == build_footman:
                if barracks_amount <= 0 or gold_amount < 600:
                    rwdA = -1


        #reward agent for number of soldiers built
        current = self.get_drts_number_of_specific_units(obs, player, footman) 
        prev = self.get_drts_number_of_specific_units(self.previous_state, player, footman) 
        rwdB = (current - prev)

        #rwd = rwdA + rwdB + rwdC
        rwd = rwdA + rwdB
        return rwd
        #print(rwd)
        #if rwd > 0: return rwd
        #else: return 0

    def get_sc2_reward(self, obs):
        #current = self.get_sc2_number_of_supply_depot(obs)
        #prev = self.get_sc2_number_of_supply_depot(self.previous_state)

        #rwdA = (current - prev)

        #current = self.get_sc2_number_of_barracks(obs)
        #prev = self.get_sc2_number_of_barracks(self.previous_state)

        #rwdB = (current - prev) * 10

        build_supply_depot = BuildUnitsGeneralizedRewardBuilder.ACTION_BUILD_SUPPLY_DEPOT
        build_barrack = BuildUnitsGeneralizedRewardBuilder.ACTION_BUILD_BARRACK
        build_marine = BuildUnitsGeneralizedRewardBuilder.ACTION_BUILD_MARINE

        rwdA = 0
        chosen_action = BuildUnitsGeneralizedRewardBuilder.LAST_CHOSEN_ACTION 
        if chosen_action > -1:
            supply_depot_amount = self.get_sc2_number_of_supply_depot(obs)
            barracks_amount = self.get_sc2_number_of_barracks(obs)
            minerals = obs.player.minerals
            if chosen_action == build_supply_depot: 
                if supply_depot_amount > 7 or minerals < 100:
                    rwdA = -1
            elif chosen_action == build_barrack:
                if supply_depot_amount <= 0 or minerals < 150:
                    rwdA = -1
            elif chosen_action == build_marine:
                if barracks_amount <= 0 or minerals < 50:
                    rwdA = -1

        current = self.get_sc2_number_of_marines(obs)
        prev = self.get_sc2_number_of_marines(self.previous_state)
        rwdB = (current - prev)

        #rwd = rwdA + rwdB + rwdC
        rwd = rwdA + rwdB
        return rwd
        #if rwd > 0: return rwd
        #else: return 0
