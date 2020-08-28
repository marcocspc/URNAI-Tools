from urnai.agents.states.abstate import StateBuilder  
from urnai.utils.constants import RTSGeneralization, Games 
import numpy as np

class CollectablesGeneralizedStatebuilder(StateBuilder):

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

    def build_state(self, obs):
        game = self.get_game(obs)
        if game == Games.DRTS:
            return self.build_drts_state(obs)
        else:
            return self.build_sc2_state(obs)

    def build_drts_state(self, obs):
        state = [] 
        if self.method == RTSGeneralization.STATE_MAP:
            state = self.build_drts_map(obs)
        elif self.method == RTSGeneralization.STATE_NON_SPATIAL:
            state = self.build_non_spatial_drts_state(obs)
        elif self.method == RTSGeneralization.STATE_BOTH: 
            state = self.build_drts_map(obs)
            state += self.build_non_spatial_drts_state(obs)


        state = list(np.asarray(state).flatten())

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

        state = list(np.asarray(state).flatten())

        
        return state

    def build_sc2_map(self, obs):
        #layer 3 is base (2 walkable area, 0 not)
        #layer 4 is units (1 friendly, 2 enemy, 16 mineral shards, 3 neutral 
        map_ = self.build_basic_sc2_map(obs)
        map_ = self.normalize_map(map_)

        return map_

    def build_basic_sc2_map(self, obs):
        map_ = np.zeros(obs.feature_minimap[0].shape)
        for y in range(len(obs.feature_minimap[3])):  
            for x in range(len(obs.feature_minimap[3][y])):  
                if obs.feature_minimap[3][y][x] == 2: map_[y][x] = 270 #drts walkable area 

        for y in range(len(obs.feature_minimap[4])):  
            for x in range(len(obs.feature_minimap[4][y])):  
                if obs.feature_minimap[4][y][x] == 1: map_[y][x] = 1 #drts 1 is peasant, 7 is archer, which one is needed for the current map 
                elif obs.feature_minimap[4][y][x] == 2: map_[y][x] = 7 #drts 1 is peasant, 7 is archer, which one is needed for the current map 
                elif obs.feature_minimap[4][y][x] == 16: map_[y][x] = 1000 #drts 1000 was chosen for me to represent virtual shards
                elif obs.feature_minimap[4][y][x] == 3: map_[y][x] = 102 #drts 102 is gold 

        return map_

    def build_drts_map(self, obs): 
        map_ = self.build_basic_drts_map(obs)


        for y in range(len(obs['collectables_map'])): 
            for x in range(len(obs['collectables_map'][y])):
                if obs['collectables_map'][y][x] == 1: 
                    map_[y][x] = 1000
                if map_[y][x] != 1000 and map_[y][x] != 7: 
                    map_[y][x] = 0

        map_ = self.normalize_map(map_)

        return map_

    def build_basic_drts_map(self, obs): 
        width = obs["map"].map_width
        height = obs["map"].map_height
        map_ = np.zeros((width, height)) 
        for tile in obs["tiles"]:
            map_[tile.y, tile.x] = tile.get_type_id()

        for unit in obs["units"]:
            map_[unit.tile.y, unit.tile.x] = int(unit.type)

        return map_

    def normalize_map(self, map_):
        map_ = (map_ - map_.min())/(map_.max() - map_.min())
        return map_

    def get_state_dim(self):
        if self.method == RTSGeneralization.STATE_MAP:
            return 64*64
            state = self.build_drts_map(obs)
