from .defeatenemies import DefeatEnemiesGeneralizedRewardBuilder 

class BuildUnitsGeneralizedRewardBuilder(DefeatEnemiesGeneralizedRewardBuilder):

    def get_drts_reward(self, obs):
        player = 0
        farm = 6
        barracks = 4
        footman = 5

        current = self.get_drts_number_of_specific_units(obs, player, farm) 
        prev = self.get_drts_number_of_specific_units(self.previous_state, player, farm) 

        rwdA = (current - prev)

        current = self.get_drts_number_of_specific_units(obs, player, barracks) 
        prev = self.get_drts_number_of_specific_units(self.previous_state, player, barracks) 

        rwdB = (current - prev)

        current = self.get_drts_number_of_specific_units(obs, player, footman) 
        prev = self.get_drts_number_of_specific_units(self.previous_state, player, footman) 

        rwdC = (current - prev)

        return (rwdA + rwdB + rwdC) * 1000

    def get_sc2_reward(self, obs):
        current = self.get_sc2_number_of_supply_depot(obs)
        prev = self.get_sc2_number_of_supply_depot(self.previous_state)

        rwdA = (current - prev)

        current = self.get_sc2_number_of_barracks(obs)
        prev = self.get_sc2_number_of_barracks(self.previous_state)

        rwdB = (current - prev)

        current = self.get_sc2_number_of_marines(obs)
        prev = self.get_sc2_number_of_marines(self.previous_state)

        rwdC = (current - prev)

        return (rwdA + rwdB + rwdC) * 1000
