from urnai.agents.states.abstate import StateBuilder  
from urnai.utils.constants import RTSGeneralization, Games 
import urnai.agents.actions.sc2 as sc2aux 
from pysc2.lib import units as sc2units
import numpy as np
from statistics import mean


class CollectablesGeneralizedStatebuilder(StateBuilder):

    def __init__(self, method=RTSGeneralization.STATE_MAP):
        self.previous_state = None
        self.method = method
        self.non_spatial_maximums = [
                RTSGeneralization.STATE_MAXIMUM_X,
                RTSGeneralization.STATE_MAXIMUM_Y,
                RTSGeneralization.STATE_MAXIMUM_NUMBER_OF_MINERAL_SHARDS,
                ]
        self.non_spatial_minimums = [0, 0, 0]
        #non-spatial is composed of
        #X distance to next mineral shard 
        #Y distance to next mineral shard 
        #number of mineral shards left
        self.non_spatial_state = [0, 0, 0]

    def get_game(self, obs):
        try:
            a = obs.feature_minimap
            return Games.SC2
        except AttributeError as ae:
            if "feature_minimap" in str(ae):
                return Games.DRTS
            else: raise

    def build_state(self, obs):
        game = self.get_game(obs)
        if game == Games.DRTS:
            state = self.build_drts_state(obs)
        else:
            state = self.build_sc2_state(obs)

        return state

    def build_drts_state(self, obs):
        state = [] 
        if self.method == RTSGeneralization.STATE_MAP:
            state = self.build_drts_map(obs)
        elif self.method == RTSGeneralization.STATE_NON_SPATIAL:
            state = self.build_non_spatial_drts_state(obs)
        elif self.method == RTSGeneralization.STATE_BOTH: 
            state = self.build_drts_map(obs)
            state += self.build_non_spatial_drts_state(obs)

        state = np.asarray(state).flatten()
        state = state.reshape((1, len(state)))

        return state

    def build_sc2_state(self, obs):
        state = [] 
        if self.method == RTSGeneralization.STATE_MAP:
            state = self.build_sc2_map(obs)
        elif self.method == RTSGeneralization.STATE_NON_SPATIAL:
            state = self.build_non_spatial_sc2_state(obs)
        elif self.method == RTSGeneralization.STATE_BOTH: 
            state = self.build_sc2_map(obs)
            state += self.build_non_spatial_sc2_state(obs)

        state = np.asarray(state).flatten()
        state = state.reshape((1, len(state)))

        
        return state

    def build_sc2_map(self, obs):
        map_ = self.build_basic_sc2_map(obs)
        map_ = self.normalize_map(map_)

        return map_

    def build_basic_sc2_map(self, obs):
        #layer 3 is base (2 walkable area, 0 not)
        #layer 4 is units (1 friendly, 2 enemy, 16 mineral shards, 3 neutral 
        map_ = np.zeros(obs.feature_minimap[0].shape)
        for y in range(len(obs.feature_minimap[3])):  
            for x in range(len(obs.feature_minimap[3][y])):  
                if obs.feature_minimap[3][y][x] == 2: map_[y][x] = 270 #drts walkable area 

        for y in range(len(obs.feature_minimap[4])):  
            for x in range(len(obs.feature_minimap[4][y])):  
                if obs.feature_minimap[4][y][x] == 1: map_[y][x] = 1 #drts 1 is peasant, 7 is archer, which one is needed for the current map 
                elif obs.feature_minimap[4][y][x] == 2: map_[y][x] = 7 #drts 1 is peasant, 7 is archer, which one is needed for the current map 
                elif obs.feature_minimap[4][y][x] == 16: map_[y][x] = 100 #drts 1000 was chosen by me to represent virtual shards
                elif obs.feature_minimap[4][y][x] == 3: map_[y][x] = 102 #drts 102 is gold 

        return map_

    def build_drts_map(self, obs): 
        map_ = self.build_basic_drts_map(obs)

        for y in range(len(obs['collectables_map'])): 
            for x in range(len(obs['collectables_map'][y])):
                if obs['collectables_map'][y][x] == 1: 
                    map_[y][x] = 100
                if map_[y][x] != 100 and map_[y][x] != 7: 
                    map_[y][x] = 0

        map_ = self.normalize_map(map_)

        return map_

    def build_basic_drts_map(self, obs): 
        width = obs["map"].map_width
        height = obs["map"].map_height
        map_ = np.zeros((width, height)) 
        for tile in obs["tiles"]:
            if int(tile.get_type_id()) == 102: #102 is gold
                map_[tile.y, tile.x] = tile.get_type_id()

        for unit in obs["units"]:
            if unit.tile is not None:
                map_[unit.tile.y, unit.tile.x] = int(unit.type)

        return map_

    def normalize_map(self, map_):
        map = (map_ - map_.min())/(map_.max() - map_.min())
        return map_

    def normalize_non_spatial_list(self):
        for i in range(len(self.non_spatial_state)):
            value = self.non_spatial_state[i]
            max_ = self.non_spatial_maximums[i]
            min_ = self.non_spatial_minimums[i]
            value = self.normalize_value(value, max_, min_) 
            self.non_spatial_state[i] = value

    def normalize_value(self, value, max_, min_=0):
        return (value - min_)/(max_ - min_) 

    def get_state_dim(self):
        if self.method == RTSGeneralization.STATE_MAP:
            return 64*64 
        elif self.method == RTSGeneralization.STATE_NON_SPATIAL:
            return len(self.non_spatial_state)

    def get_sc2_marine_mean(self, obs):
        xs = []
        ys = []
         
        for unit in sc2aux.get_units_by_type(obs, sc2units.Terran.Marine):
            xs.append(unit.x)
            ys.append(unit.y)

        x_mean = mean(xs) 
        y_mean = mean(ys) 

        return x_mean, y_mean

    def get_closest_sc2_mineral_shard_x_y(self, obs):
        x_closest_distance = RTSGeneralization.STATE_MAXIMUM_X 
        y_closest_distance = RTSGeneralization.STATE_MAXIMUM_Y 
        x, y = self.get_sc2_marine_mean(obs)
        for mineral_shard_y in range(len(obs.feature_minimap[4])):
            for mineral_shard_x in range(len(obs.feature_minimap[4][0])):
                mineral_shard = (obs.feature_minimap[4][y][x] == 16)
                if mineral_shard: 
                    x_dist = x - mineral_shard_x 
                    y_dist = y - mineral_shard_y 
                    if x_dist < x_closest_dist: x_closest_distance = x_dist
                    if y_dist < y_closest_dist: y_closest_distance = y_dist

        return x_closest_distance, y_closest_distance

    def build_non_spatial_sc2_state(self, obs):
        x, y = self.get_closest_sc2_mineral_shard_x_y(obs)
        #position 0: distance x to closest shard 
        self.non_spatial_state[0] = x
        #position 1: distance y to closest shard
        self.non_spatial_state[1] = y
        #position 2: number of remaining shards
        self.non_spatial_state[2] = np.count_nonzero(obs.feature_minimap[4] == 16)
        self.normalize_non_spatial_list() 
        #spatial values need a second normalization because the value can
        #be negative, so they are summed with 1 and then normalized 
        #again 
        self.non_spatial_state[0] = self.normalize_value(self.non_spatial_state[0] + 1, 2)
        self.non_spatial_state[1] = self.normalize_value(self.non_spatial_state[1] + 1, 2)
        return self.non_spatial_state

    def build_non_spatial_drts_state(self, obs):
        x, y = self.get_closest_drts_mineral_shard_x_y(obs)
        #position 0: distance x to closest shard 
        self.non_spatial_state[0] = x
        #position 1: distance y to closest shard
        self.non_spatial_state[1] = y
        #position 2: number of remaining shards
        self.non_spatial_state[2] = np.count_nonzero(obs['collectables_map'] == 1)
        self.normalize_non_spatial_list() 
        #spatial values need a second normalization because the value can
        #be negative, so they are summed with 1 and then normalized 
        #again 
        self.non_spatial_state[0] = self.normalize_value(self.non_spatial_state[0] + 1, 2)
        self.non_spatial_state[1] = self.normalize_value(self.non_spatial_state[1] + 1, 2)
        return self.non_spatial_state

    def get_closest_drts_mineral_shard_x_y(self, obs):
        x_closest_distance = RTSGeneralization.STATE_MAXIMUM_X 
        y_closest_distance = RTSGeneralization.STATE_MAXIMUM_Y 
        x, y = self.get_drts_army_mean(obs)
        for mineral_shard_y in range(len(obs['collectables_map'])):
            for mineral_shard_x in range(len(obs['collectables_map'][0])):
                x_dist = x - mineral_shard_x 
                y_dist = y - mineral_shard_y 
                if x_dist < x_closest_distance: x_closest_distance = x_dist
                if y_dist < y_closest_distance: y_closest_distance = y_dist

        return x_closest_distance, y_closest_distance

    def get_drts_army_mean(self, obs):
        xs = []
        ys = []
         
        for unit in self.get_drts_player_units(obs, 0):
            xs.append(unit.tile.x)
            ys.append(unit.tile.y)

        x_mean = mean(xs) 
        y_mean = mean(ys) 

        return x_mean, y_mean

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
