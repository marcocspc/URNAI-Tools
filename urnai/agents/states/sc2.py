import math
import numpy as np
import scipy.misc
from matplotlib import colors
from matplotlib import pyplot as plt
from .abstate import StateBuilder
from pysc2.lib import actions, features, units
from agents.actions.sc2 import *
from pysc2.env import sc2_env


class Simple64State(StateBuilder):

    def __init__(self):
        self.reduction_factor = 4

        self._state_size = 22 + (64/self.reduction_factor)**2
        self.player_race = 0

    def build_state(self, obs):
        if obs.game_loop[0] == 0:

            commandcenter = get_my_units_by_type(obs, units.Terran.CommandCenter)
            nexus = get_my_units_by_type(obs, units.Protoss.Nexus)
            hatchery = get_my_units_by_type(obs, units.Zerg.Hatchery)
            if len(commandcenter)>0: 
                townhall = commandcenter[0]
                self.player_race = sc2_env.Race.terran
            if len(nexus)>0:
                townhall = nexus[0]
                self.player_race = sc2_env.Race.protoss
            if len(hatchery)>0:
                townhall = hatchery[0]
                self.player_race = sc2_env.Race.zerg

            self.base_top_left = (townhall.x < 32)

        new_state = []
        new_state.append(obs.player.minerals)
        new_state.append(obs.player.vespene)
        new_state.append(obs.player.food_cap)
        new_state.append(obs.player.food_used)
        new_state.append(obs.player.food_army)
        new_state.append(obs.player.food_workers)
        new_state.append(obs.player.food_cap - obs.player.food_used)
        new_state.append(obs.player.army_count)
        new_state.append(obs.player.idle_worker_count)

        if self.player_race == sc2_env.Race.terran:
            new_state.append(get_units_amount(obs, units.Terran.CommandCenter)+
                            get_units_amount(obs, units.Terran.OrbitalCommand)+
                            get_units_amount(obs, units.Terran.PlanetaryFortress))
            new_state.append(get_units_amount(obs, units.Terran.SupplyDepot))
            new_state.append(get_units_amount(obs, units.Terran.Refinery))
            new_state.append(get_units_amount(obs, units.Terran.EngineeringBay))
            new_state.append(get_units_amount(obs, units.Terran.Armory))
            new_state.append(get_units_amount(obs, units.Terran.MissileTurret))
            new_state.append(get_units_amount(obs, units.Terran.SensorTower))
            new_state.append(get_units_amount(obs, units.Terran.Bunker))
            new_state.append(get_units_amount(obs, units.Terran.FusionCore))
            new_state.append(get_units_amount(obs, units.Terran.GhostAcademy))
            new_state.append(get_units_amount(obs, units.Terran.Barracks))
            new_state.append(get_units_amount(obs, units.Terran.Factory))
            new_state.append(get_units_amount(obs, units.Terran.Starport))

        elif self.player_race == sc2_env.Race.protoss:
            new_state.append(get_units_amount(obs, units.Protoss.Nexus))
            new_state.append(get_units_amount(obs, units.Protoss.Pylon))
            new_state.append(get_units_amount(obs, units.Protoss.Assimilator))
            new_state.append(get_units_amount(obs, units.Protoss.Forge))
            new_state.append(get_units_amount(obs, units.Protoss.Gateway))
            new_state.append(get_units_amount(obs, units.Protoss.CyberneticsCore))
            new_state.append(get_units_amount(obs, units.Protoss.PhotonCannon))
            new_state.append(get_units_amount(obs, units.Protoss.RoboticsFacility))
            new_state.append(get_units_amount(obs, units.Protoss.Stargate))
            new_state.append(get_units_amount(obs, units.Protoss.TwilightCouncil))
            new_state.append(get_units_amount(obs, units.Protoss.RoboticsBay))
            new_state.append(get_units_amount(obs, units.Protoss.TemplarArchive))
            new_state.append(get_units_amount(obs, units.Protoss.DarkShrine))

        # TO DO: Append observations for zergs
        #elif self.player_race == sc2_env.Race.zerg:

        m0 = obs.feature_minimap[0]     # Feature layer of the map's terrain (elevation and shape)
        m0 = m0/10
        m1 = obs.feature_minimap[2]     # Feature layer of creep in the minimap (generally will be quite empty, especially on games without zergs hehe)
        m1[m1 == 1] = 16                # Transforming creep info from 1 to 8 in the map (makes it more visible)
        m2 = obs.feature_minimap[4]     # Feature layer of all visible units (neutral, friendly and enemy) on the minimap
        m2[m2 == 1] = 128               # Transforming own units from 1 to 64 for visibility
        m2[m2 == 3] = 64                # Transforming enemy units from 3 to 32 for visibility
        m2[m2 == 2] = 64
        m2[m2 == 16] = 32               # Transforming mineral fields and geysers from 16 to 32 for visibility               
        
        combined_minimap = np.where(m2 != 0, m2, m1)                                    # Overlaying the m2(units) map over the m1(creep) map
        combined_minimap = np.where(combined_minimap != 0, combined_minimap, m0)        # Overlaying the combined m1 and m2 map over the m0(terrain) map
        # combined_minimap = m0+m1+m2
        combined_minimap = np.array(combined_minimap)                                   # Tranforming combined_minimap into a np array so tensor flow can interpret it
        combined_minimap = combined_minimap/combined_minimap.max()                      # Normalizing between 0 and 1

        # Lowering the featuremap's resolution
        lowered_minimap = lower_featuremap_resolution(combined_minimap, self.reduction_factor)      #featuremap and reduction factor, if rf = 4 a 64x64 map will be transformed into a 16x16 map
        
        # Rotating observation depending on Agent's location on the map so that we get a consistent, generalized, observation
        if not self.base_top_left: 
            lowered_minimap = np.rot90(lowered_minimap, 2)
        
        # Displaying the agent's vision in a heatmap using matplotlib every 200 steps (just for debug purpuses, probably will be removed later)
        # if (obs.game_loop[0]/16)%200 == 0:
        #     # Displaying Agent's vision
        #     #norm = colors.Normalize()
        #     plt.imshow(lowered_minimap, interpolation='nearest')
        #     plt.show()

        new_state.extend(lowered_minimap.flatten())   
        final_state = np.array(new_state)
        final_state = np.expand_dims(final_state, axis=0)

        return final_state


    def get_state_dim(self):
        return self._state_size


class Simple64State_1(StateBuilder):
    def build_state(self, obs):
        
        if obs.game_loop[0] == 0:
            command_center = get_my_units_by_type(obs, units.Terran.CommandCenter)[0]
            self.base_top_left = (command_center.x < 32)

        # Whether or not our supply depot was built
        supply_depot_count = get_units_amount(obs, units.Terran.SupplyDepot)

        # Whether or not our barracks were built
        barracks_count = get_units_amount(obs, units.Terran.Barracks)

        army_count = len(select_army(obs, sc2_env.Race.terran))

        # The supply limit
        supply_limit = obs.player[4]
        # The army supply
        supply_army = obs.player[5]
        # Free supply
        supply_free = get_free_supply(obs)

        # Defining our state, considering our enemies' positions.
        current_state = np.zeros(22)
        current_state[0] = supply_depot_count
        current_state[1] = barracks_count
        current_state[2] = supply_limit
        current_state[3] = supply_army
        current_state[4] = supply_free
        current_state[5] = army_count
        

        # Insteading of making a vector for all coordnates on the map, we'll discretize our enemy space
        # and use a 16x16 grid to store enemy positions by marking a square as 1 if there's any enemy on it.
        hot_squares = np.zeros(16)
        enemy_units = [unit for unit in obs.raw_units if unit.alliance == features.PlayerRelative.ENEMY]
        
        for i in range(0, len(enemy_units)):
            y = int(math.ceil((enemy_units[i].x + 1) / 16))
            x = int(math.ceil((enemy_units[i].y + 1) / 16))

            hot_squares[((y - 1) * 4) + (x - 1)] = 1

        if not self.base_top_left:
            hot_squares = hot_squares[::-1]

        for i in range(0, 16):
            # Adds 4 to account for supply_depot_count, barracks_count, supply_limit and army_supply
            current_state[i + 6] = hot_squares[i]
        
        current_state = np.expand_dims(current_state, axis=0)
        return current_state


    def get_state_dim(self):
        return 22