import random
import math
import os.path
import numpy as np
from .base.sc2_abagent import SC2Agent
from pysc2.lib import actions, features
from pysc2.agents.base_agent import BaseAgent

_STATE_SIZE = 12

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_PLAYER_ID = features.SCREEN_FEATURES.player_id.index

_PLAYER_SELF = 1
_PLAYER_HOSTILE = 4
_ARMY_SUPPLY = 5

_TERRAN_COMMANDCENTER = 18
_TERRAN_SUPPLY_DEPOT = 19
_TERRAN_BARRACKS = 21
_TERRAN_SCV = 45
_NEUTRAL_MINERAL_FIELD = 341


class TerranAgentSparse(SC2Agent):
    
    def __init__(self, action_wrapper):
        super(TerranAgentSparse, self).__init__(action_wrapper)

        self.cc_y = None
        self.cc_x = None


    def get_reward(self, obs, reward, done):
        if not done:
            return 0
        return reward


    def build_state(self, obs):
        player_y, player_x = (obs.feature_minimap[_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
        self.base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0

        # Setting up the player's base position
        unit_type = obs.feature_screen[_UNIT_TYPE]
        cc_y, cc_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()
        cc_count = 1 if cc_y.any() else 0

        depot_y, depot_x = (unit_type == _TERRAN_SUPPLY_DEPOT).nonzero()
        supply_depot_count = int(round(len(depot_y) / 69))

        barracks_y, barracks_x = (unit_type == _TERRAN_BARRACKS).nonzero()
        barracks_count = int(round(len(barracks_y) / 137))

        # Defining our state:
        ## cc_count, supply_depot_count, barracks_count, player.army_supply
        # Enemies positions (4), Friendly army positions (4)
        new_state = np.zeros(_STATE_SIZE)
        new_state[0] = cc_count
        new_state[1] = supply_depot_count
        new_state[2] = barracks_count
        new_state[3] = obs.player[_ARMY_SUPPLY]

        # Dividing our minimap into a 4x4 grid of cells and marking cells as 1 if it
        # contains any friendly army units. If the base is at the bottom right, we invert the
        # quadrants so that it's seen from the perspective of a top-left base
        green_squares = np.zeros(4)
        friendly_y, friendly_x = (obs.feature_minimap[_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()

        for i in range(0, len(friendly_y)):
            y = int(math.ceil((friendly_y[i] + 1) / 32))
            x = int(math.ceil((friendly_x[i] + 1) / 32))

            green_squares[((y - 1) * 2) + (x - 1)] = 1

        if not self.base_top_left:
            green_squares = green_squares[::-1]

        for i in range(0, 4):
            new_state[i + 4] = green_squares[i]

        # Adding enemy units locations to our state the same way we did with the friendly army.
        hot_squares = np.zeros(4)
        enemy_y, enemy_x = (obs.feature_minimap[_PLAYER_RELATIVE] == _PLAYER_HOSTILE).nonzero()
        for i in range(0, len(enemy_y)):
            y = int(math.ceil((enemy_y[i] + 1) / 32))
            x = int(math.ceil((enemy_x[i] + 1) / 32))

            hot_squares[((y - 1) * 2) + (x - 1)] = 1

        if not self.base_top_left:
            hot_squares = hot_squares[::-1]

        for i in range(0, 4):
            new_state[i + 8] = hot_squares[i]

        new_state = np.expand_dims(new_state, axis=0)
        return new_state


    def get_state_dim(self):
        return _STATE_SIZE
