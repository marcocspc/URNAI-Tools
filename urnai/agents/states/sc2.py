import math
import numpy as np
import scipy.misc
from matplotlib import colors
from matplotlib import pyplot as plt
from .abstate import StateBuilder
from pysc2.lib import actions, features, units
from agents.actions.sc2 import *
from pysc2.env import sc2_env
from utils.image import *


class Simple64State(StateBuilder):

    def __init__(self, reduction_factor=4):
        self.reduction_factor = reduction_factor

        self._state_size = int(22 + (64/self.reduction_factor)**2)
        self.player_race = 0

    def build_state(self, obs):
        if obs.game_loop[0] < 80 and self.base_top_left == None:

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
        new_state.append(obs.player.minerals/6000)
        new_state.append(obs.player.vespene/4000)
        new_state.append(obs.player.food_cap/200)
        new_state.append(obs.player.food_used/200)
        new_state.append(obs.player.food_army/200)
        new_state.append(obs.player.food_workers/200)
        new_state.append((obs.player.food_cap - obs.player.food_used)/200)
        new_state.append(obs.player.army_count/200)
        new_state.append(obs.player.idle_worker_count/200)

        if self.player_race == sc2_env.Race.terran:
            new_state.append(get_units_amount(obs, units.Terran.CommandCenter)+
                            get_units_amount(obs, units.Terran.OrbitalCommand)+
                            get_units_amount(obs, units.Terran.PlanetaryFortress)/2)
            new_state.append(get_units_amount(obs, units.Terran.SupplyDepot)/8)
            new_state.append(get_units_amount(obs, units.Terran.Refinery)/4)
            new_state.append(get_units_amount(obs, units.Terran.EngineeringBay))
            new_state.append(get_units_amount(obs, units.Terran.Armory))
            new_state.append(get_units_amount(obs, units.Terran.MissileTurret)/8)
            new_state.append(get_units_amount(obs, units.Terran.SensorTower)/3)
            new_state.append(get_units_amount(obs, units.Terran.Bunker)/5)
            new_state.append(get_units_amount(obs, units.Terran.FusionCore))
            new_state.append(get_units_amount(obs, units.Terran.GhostAcademy))
            new_state.append(get_units_amount(obs, units.Terran.Barracks)/3)
            new_state.append(get_units_amount(obs, units.Terran.Factory)/2)
            new_state.append(get_units_amount(obs, units.Terran.Starport)/2)

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
            
        elif self.player_race == sc2_env.Race.zerg:
            new_state.append(get_units_amount(obs, units.Zerg.BanelingNest))
            new_state.append(get_units_amount(obs, units.Zerg.EvolutionChamber))
            new_state.append(get_units_amount(obs, units.Zerg.Extractor))
            new_state.append(get_units_amount(obs, units.Zerg.Hatchery))
            new_state.append(get_units_amount(obs, units.Zerg.HydraliskDen))
            new_state.append(get_units_amount(obs, units.Zerg.InfestationPit))
            new_state.append(get_units_amount(obs, units.Zerg.LurkerDen))
            new_state.append(get_units_amount(obs, units.Zerg.NydusNetwork))
            new_state.append(get_units_amount(obs, units.Zerg.RoachWarren))
            new_state.append(get_units_amount(obs, units.Zerg.SpawningPool))
            new_state.append(get_units_amount(obs, units.Zerg.SpineCrawler))
            new_state.append(get_units_amount(obs, units.Zerg.Spire))
            new_state.append(get_units_amount(obs, units.Zerg.SporeCrawler))


        m0 = obs.feature_minimap[0]     # Feature layer of the map's terrain (elevation and shape)
        m0 = m0/10

        m1 = obs.feature_minimap[2]     # Feature layer of creep in the minimap (generally will be quite empty, especially on games without zergs hehe)
        m1[m1 == 1] = 16                # Transforming creep info from 1 to 16 in the map (makes it more visible)

        m2 = obs.feature_minimap[4]     # Feature layer of all visible units (neutral, friendly and enemy) on the minimap
        m2[m2 == 1] = 128               # Transforming own units from 1 to 128 for visibility
        m2[m2 == 3] = 64                # Transforming enemy units from 3 to 64 for visibility
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
        
        new_state.extend(lowered_minimap.flatten())   
        final_state = np.array(new_state)
        final_state = np.expand_dims(final_state, axis=0)


        # Displaying the agent's vision in a heatmap using matplotlib every 200 steps (just for debug purpuses, probably will be removed later)
        # if (obs.game_loop[0]/16)%300 == 0:
        #     # Displaying Agent's vision
        #     #norm = colors.Normalize()
        #     plt.figure()
        #     #plt.imshow(np.array(obs.feature_minimap[0]))
        #     #plt.imshow(np.array(obs.feature_minimap[2]))
        #     #plt.imshow(np.array(obs.feature_minimap[4]))
        #     #plt.imshow(np.array(combined_minimap))
        #     plt.imshow(np.array(mtest))
        #     plt.show()

        return final_state


    def get_state_dim(self):
        return self._state_size

    
class Simple64StateFullRes(StateBuilder):

    def __init__(self, reduction_factor=2):
        #self._state_size = 1957

        self.reduction_factor = reduction_factor
        self._state_size = int(22 + (44/self.reduction_factor)**2)
        self.player_race = 0

    def build_state(self, obs):
        if obs.game_loop[0] < 80 and self.base_top_left == None:

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
        new_state.append(obs.player.minerals/6000)
        new_state.append(obs.player.vespene/4000)
        new_state.append(obs.player.food_cap/200)
        new_state.append(obs.player.food_used/200)
        new_state.append(obs.player.food_army/200)
        new_state.append(obs.player.food_workers/200)
        new_state.append((obs.player.food_cap - obs.player.food_used)/200)
        new_state.append(obs.player.army_count/200)
        new_state.append(obs.player.idle_worker_count/200)

        if self.player_race == sc2_env.Race.terran:
            new_state.append(get_units_amount(obs, units.Terran.CommandCenter)+
                            get_units_amount(obs, units.Terran.OrbitalCommand)+
                            get_units_amount(obs, units.Terran.PlanetaryFortress)/2)
            new_state.append(get_units_amount(obs, units.Terran.SupplyDepot)/8)
            new_state.append(get_units_amount(obs, units.Terran.Refinery)/4)
            new_state.append(get_units_amount(obs, units.Terran.EngineeringBay))
            new_state.append(get_units_amount(obs, units.Terran.Armory))
            new_state.append(get_units_amount(obs, units.Terran.MissileTurret)/8)
            new_state.append(get_units_amount(obs, units.Terran.SensorTower)/3)
            new_state.append(get_units_amount(obs, units.Terran.Bunker)/5)
            new_state.append(get_units_amount(obs, units.Terran.FusionCore))
            new_state.append(get_units_amount(obs, units.Terran.GhostAcademy))
            new_state.append(get_units_amount(obs, units.Terran.Barracks)/3)
            new_state.append(get_units_amount(obs, units.Terran.Factory)/2)
            new_state.append(get_units_amount(obs, units.Terran.Starport)/2)

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
            
        elif self.player_race == sc2_env.Race.zerg:
            new_state.append(get_units_amount(obs, units.Zerg.BanelingNest))
            new_state.append(get_units_amount(obs, units.Zerg.EvolutionChamber))
            new_state.append(get_units_amount(obs, units.Zerg.Extractor))
            new_state.append(get_units_amount(obs, units.Zerg.Hatchery))
            new_state.append(get_units_amount(obs, units.Zerg.HydraliskDen))
            new_state.append(get_units_amount(obs, units.Zerg.InfestationPit))
            new_state.append(get_units_amount(obs, units.Zerg.LurkerDen))
            new_state.append(get_units_amount(obs, units.Zerg.NydusNetwork))
            new_state.append(get_units_amount(obs, units.Zerg.RoachWarren))
            new_state.append(get_units_amount(obs, units.Zerg.SpawningPool))
            new_state.append(get_units_amount(obs, units.Zerg.SpineCrawler))
            new_state.append(get_units_amount(obs, units.Zerg.Spire))
            new_state.append(get_units_amount(obs, units.Zerg.SporeCrawler))


        m0 = obs.feature_minimap[0]     # Feature layer of the map's terrain (elevation and shape)
        m0 = m0/10

        m1 = obs.feature_minimap[2]     # Feature layer of creep in the minimap (generally will be quite empty, especially on games without zergs hehe)
        m1[m1 == 1] = 16                # Transforming creep info from 1 to 16 in the map (makes it more visible)

        m2 = obs.feature_minimap[4]     # Feature layer of all visible units (neutral, friendly and enemy) on the minimap
        m2[m2 == 1] = 128               # Transforming own units from 1 to 128 for visibility
        m2[m2 == 3] = 64                # Transforming enemy units from 3 to 64 for visibility
        m2[m2 == 2] = 64
        m2[m2 == 16] = 32               # Transforming mineral fields and geysers from 16 to 32 for visibility

        
        combined_minimap = np.where(m2 != 0, m2, m1)                                    # Overlaying the m2(units) map over the m1(creep) map
        combined_minimap = np.where(combined_minimap != 0, combined_minimap, m0)        # Overlaying the combined m1 and m2 map over the m0(terrain) map
        # combined_minimap = m0+m1+m2
        combined_minimap = trim_feature_minimap(combined_minimap)
        combined_minimap = np.array(combined_minimap)                                   # Tranforming combined_minimap into a np array so tensor flow can interpret it
        combined_minimap = combined_minimap/combined_minimap.max()                      # Normalizing between 0 and 1

        # Lowering the featuremap's resolution
        lowered_minimap = lower_featuremap_resolution(combined_minimap, self.reduction_factor)
        
        # Rotating observation depending on Agent's location on the map so that we get a consistent, generalized, observation
        if not self.base_top_left: 
            lowered_minimap = np.rot90(lowered_minimap, 2)
        
        new_state.extend(lowered_minimap.flatten())   
        final_state = np.array(new_state)
        final_state = np.expand_dims(final_state, axis=0)


        # Displaying the agent's vision in a heatmap using matplotlib every 200 steps (just for debug purpuses, probably will be removed later)
        # if (obs.game_loop[0]/16)%50 == 0:
        #     # Displaying Agent's vision
        #     plt.figure()
        #     plt.imshow(lowered_minimap)
            
        #     plt.show()

        return final_state


    def get_state_dim(self):
        return self._state_size


class Simple64GridState(StateBuilder):
    def __init__(self, grid_size=4):

        self.grid_size = grid_size
        self._state_size = int(22 + 2*(self.grid_size**2))
        self.player_race = 0
        self.base_top_left = None

    def build_state(self, obs):
        if obs.game_loop[0] < 80 and self.base_top_left == None:

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
        new_state.append(obs.player.minerals/6000)
        new_state.append(obs.player.vespene/4000)
        new_state.append(obs.player.food_cap/200)
        new_state.append(obs.player.food_used/200)
        new_state.append(obs.player.food_army/200)
        new_state.append(obs.player.food_workers/200)
        new_state.append((obs.player.food_cap - obs.player.food_used)/200)
        new_state.append(obs.player.army_count/200)
        new_state.append(obs.player.idle_worker_count/200)

        if self.player_race == sc2_env.Race.terran:
            new_state.append(get_units_amount(obs, units.Terran.CommandCenter)+
                            get_units_amount(obs, units.Terran.OrbitalCommand)+
                            get_units_amount(obs, units.Terran.PlanetaryFortress)/2)
            new_state.append(get_units_amount(obs, units.Terran.SupplyDepot)/8)
            new_state.append(get_units_amount(obs, units.Terran.Refinery)/4)
            new_state.append(get_units_amount(obs, units.Terran.EngineeringBay))
            new_state.append(get_units_amount(obs, units.Terran.Armory))
            new_state.append(get_units_amount(obs, units.Terran.MissileTurret)/8)
            new_state.append(get_units_amount(obs, units.Terran.SensorTower)/3)
            new_state.append(get_units_amount(obs, units.Terran.Bunker)/5)
            new_state.append(get_units_amount(obs, units.Terran.FusionCore))
            new_state.append(get_units_amount(obs, units.Terran.GhostAcademy))
            new_state.append(get_units_amount(obs, units.Terran.Barracks)/3)
            new_state.append(get_units_amount(obs, units.Terran.Factory)/2)
            new_state.append(get_units_amount(obs, units.Terran.Starport)/2)

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
            
        elif self.player_race == sc2_env.Race.zerg:
            new_state.append(get_units_amount(obs, units.Zerg.BanelingNest))
            new_state.append(get_units_amount(obs, units.Zerg.EvolutionChamber))
            new_state.append(get_units_amount(obs, units.Zerg.Extractor))
            new_state.append(get_units_amount(obs, units.Zerg.Hatchery))
            new_state.append(get_units_amount(obs, units.Zerg.HydraliskDen))
            new_state.append(get_units_amount(obs, units.Zerg.InfestationPit))
            new_state.append(get_units_amount(obs, units.Zerg.LurkerDen))
            new_state.append(get_units_amount(obs, units.Zerg.NydusNetwork))
            new_state.append(get_units_amount(obs, units.Zerg.RoachWarren))
            new_state.append(get_units_amount(obs, units.Zerg.SpawningPool))
            new_state.append(get_units_amount(obs, units.Zerg.SpineCrawler))
            new_state.append(get_units_amount(obs, units.Zerg.Spire))
            new_state.append(get_units_amount(obs, units.Zerg.SporeCrawler))     

        # Insteading of making a vector for all coordnates on the map, we'll discretize our enemy space
        # and use a 4x4 grid to store enemy positions by marking a square as 1 if there's any enemy on it.
        # enemy_grid = np.zeros((4,4))
        # player_grid = np.zeros((4,4))

        # enemy_units = [unit for unit in obs.raw_units if unit.alliance == features.PlayerRelative.ENEMY]
        # player_units = [unit for unit in obs.raw_units if unit.alliance == features.PlayerRelative.SELF]
        
        # for i in range(0, len(enemy_units)):
        #     y = int(math.ceil((enemy_units[i].x + 1) / 16))
        #     x = int(math.ceil((enemy_units[i].y + 1) / 16))
        #     #enemy_grid[((y - 1) * 4) + (x - 1)] += 1
        #     enemy_grid[x-1][y-1] += 1

        # for i in range(0, len(player_units)):
        #     y = int(math.ceil((player_units[i].x + 1) / 16))
        #     x = int(math.ceil((player_units[i].y + 1) / 16))
        #     #enemy_grid[((y - 1) * 4) + (x - 1)] += 1
        #     player_grid[x-1][y-1] += 1

        # if not self.base_top_left:
        #     enemy_grid = np.rot90(enemy_grid, 2)
        #     player_grid = np.rot90(player_grid, 2)
        #     #enemy_grid = enemy_grid[::-1]

        enemy_grid = np.zeros((self.grid_size,self.grid_size))
        player_grid = np.zeros((self.grid_size,self.grid_size))

        enemy_units = [unit for unit in obs.raw_units if unit.alliance == features.PlayerRelative.ENEMY]
        player_units = [unit for unit in obs.raw_units if unit.alliance == features.PlayerRelative.SELF]
        
        for i in range(0, len(enemy_units)):
            y = int(math.ceil((enemy_units[i].x + 1) / 64/self.grid_size))
            x = int(math.ceil((enemy_units[i].y + 1) / 64/self.grid_size))
            #enemy_grid[((y - 1) * 4) + (x - 1)] += 1
            enemy_grid[x-1][y-1] += 1

        for i in range(0, len(player_units)):
            y = int(math.ceil((player_units[i].x + 1) / (64/self.grid_size)))
            x = int(math.ceil((player_units[i].y + 1) / (64/self.grid_size)))
            #enemy_grid[((y - 1) * 4) + (x - 1)] += 1
            player_grid[x-1][y-1] += 1

        if not self.base_top_left:
            enemy_grid = np.rot90(enemy_grid, 2)
            player_grid = np.rot90(player_grid, 2)
            #enemy_grid = enemy_grid[::-1]

        # Normalizing the values to always be between 0 and 1 (since the max amount of units in SC2 is 200)
        enemy_grid = enemy_grid/200
        player_grid = player_grid/200
        
        new_state.extend(enemy_grid.flatten())
        new_state.extend(player_grid.flatten())
        final_state = np.array(new_state)
        final_state = np.expand_dims(new_state, axis=0)
        return final_state

    def get_state_dim(self):
        return self._state_size


class Simple64GridState_SimpleTerran(StateBuilder):
    def __init__(self, grid_size=4):

        self.grid_size = grid_size
        self._state_size = int(12 + 2*(self.grid_size**2))
        self.player_race = 0
        self.base_top_left = None

    def build_state(self, obs):
        if obs.game_loop[0] < 80 and self.base_top_left == None:
            commandcenter = get_my_units_by_type(obs, units.Terran.CommandCenter)
            townhall = commandcenter[0]
            self.player_race = sc2_env.Race.terran
            self.base_top_left = (townhall.x < 32)

        new_state = []
        new_state.append(obs.player.minerals/6000)
        new_state.append(obs.player.vespene/6000)
        new_state.append(obs.player.food_cap/200)
        new_state.append(obs.player.food_used/200)
        new_state.append(obs.player.food_army/200)
        new_state.append(obs.player.idle_worker_count/200)

        if self.player_race == sc2_env.Race.terran:
            new_state.append(get_units_amount(obs, units.Terran.CommandCenter)+
                            get_units_amount(obs, units.Terran.OrbitalCommand)+
                            get_units_amount(obs, units.Terran.PlanetaryFortress)/10)
            new_state.append(get_units_amount(obs, units.Terran.SupplyDepot)/10)
            new_state.append(get_units_amount(obs, units.Terran.Refinery)/10)
            new_state.append(get_units_amount(obs, units.Terran.Barracks)/10)
            new_state.append(get_units_amount(obs, units.Terran.Factory)/10)
            new_state.append(get_units_amount(obs, units.Terran.Starport)/10)  


        enemy_grid = np.zeros((self.grid_size,self.grid_size))
        player_grid = np.zeros((self.grid_size,self.grid_size))

        enemy_units = [unit for unit in obs.raw_units if unit.alliance == features.PlayerRelative.ENEMY]
        player_units = [unit for unit in obs.raw_units if unit.alliance == features.PlayerRelative.SELF]
        
        for i in range(0, len(enemy_units)):
            y = int(math.ceil((enemy_units[i].x + 1) / 64/self.grid_size))
            x = int(math.ceil((enemy_units[i].y + 1) / 64/self.grid_size))
            enemy_grid[x-1][y-1] += 1

        for i in range(0, len(player_units)):
            y = int(math.ceil((player_units[i].x + 1) / (64/self.grid_size)))
            x = int(math.ceil((player_units[i].y + 1) / (64/self.grid_size)))
            player_grid[x-1][y-1] += 1

        if not self.base_top_left:
            enemy_grid = np.rot90(enemy_grid, 2)
            player_grid = np.rot90(player_grid, 2)

        # Normalizing the values to always be between 0 and 1 (since the max amount of units in SC2 is 200)
        enemy_grid = enemy_grid/200
        player_grid = player_grid/200
        
        new_state.extend(enemy_grid.flatten())
        new_state.extend(player_grid.flatten())
        final_state = np.array(new_state)
        final_state = np.expand_dims(new_state, axis=0)
        return final_state

    def get_state_dim(self):
        return self._state_size


def trim_feature_minimap(feature_minimap):
    feature_minimap = np.delete(feature_minimap, np.s_[0:12:1], 0)
    feature_minimap = np.delete(feature_minimap, np.s_[44:63:1], 0)
    feature_minimap = np.delete(feature_minimap, np.s_[0:8:1], 1)
    feature_minimap = np.delete(feature_minimap, np.s_[44:63:1], 1)
    return feature_minimap