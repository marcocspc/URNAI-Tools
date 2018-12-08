import random
import math
import os.path
import numpy as np
from pysc2.lib import actions, features
from pysc2.agents import base_agent

# Defining constants for our agent
_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_BUILD_SUPPLY_DEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_PLAYER_ID = features.SCREEN_FEATURES.player_id.index

_PLAYER_SELF = 1
_PLAYER_HOSTILE = 4

_TERRAN_COMMANDCENTER = 18
_TERRAN_SCV = 45
_TERRAN_SUPPLY_DEPOT = 19
_TERRAN_BARRACKS = 21

_NOT_QUEUED = [0]
_QUEUED = [1]

# Actions constants. These are the actions our agent will try to use.
ACTION_DO_NOTHING = 'donothing'
ACTION_SELECT_SCV = 'selectscv'
ACTION_BUILD_SUPPLY_DEPOT = 'buildsupplydepot'
ACTION_BUILD_BARRACKS = 'buildbarracks'
ACTION_SELECT_BARRACKS = 'selectbarracks'
ACTION_BUILD_MARINE = 'buildmarine'
ACTION_SELECT_ARMY = 'selectarmy'
ACTION_ATTACK = 'attack'

smart_actions = [
    ACTION_DO_NOTHING,
    ACTION_SELECT_SCV,
    ACTION_BUILD_SUPPLY_DEPOT,
    ACTION_BUILD_BARRACKS,
    ACTION_SELECT_BARRACKS,
    ACTION_BUILD_MARINE,
    ACTION_SELECT_ARMY,
]

# Adding ACTION_ATTACK smart action to 16 coordinate combinations on the map. We choose to pick only 16
# because we're using marines to attack, and their range is big enough that we only need a 4x4 grid to cover
# the entire map. The coordinates are shifted by (8, 8) to cover the middle of a 'cell'.
for mm_x in range(0, 64):
    for mm_y in range(0, 64):
        if (mm_x + 1) % 16 == 0 and (mm_y + 1) % 16 == 0:
            smart_actions.append(ACTION_ATTACK + '_' + str(mm_x - 8) + '_' + str(mm_y - 8))


## Defining our agent's rewards
KILL_UNIT_REWARD = 0.2
KILL_BUILDING_REWARD = 0.5


def get_actions():
    return list(range(len(smart_actions)))


# Defining our agent's class
class TerranAgent(base_agent.BaseAgent):

    def __init__(self, rl_model):
        super(TerranAgent, self).__init__()

        # Setting up our Q-Learning to our agent
        self.rl_model = rl_model

        # Properties to track the change of values used in our reward system
        self.previous_killed_unit_score = 0
        self.previous_killed_building_score = 0

        self.previous_action = None
        self.previous_state = None

        self.rl_model.load()


    def step(self, obs):
        super(TerranAgent, self).step(obs)

        if obs.last():
            self.rl_model.save()

        # Setting our base position
        player_y, player_x = (obs.observation.feature_minimap[_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
        self.base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0
        

        ## Defining our state and calculating the reward
        unit_type = obs.observation.feature_screen[_UNIT_TYPE]

        depot_y, depot_x = (unit_type == _TERRAN_SUPPLY_DEPOT).nonzero()    # Whether or not our supply depot was built
        supply_depot_count = 1 if depot_y.any() else 0                    

        barracks_y, barracks_x = (unit_type == _TERRAN_BARRACKS).nonzero()  # Whether or not our barracks were built
        barracks_count = 1 if barracks_y.any else 0

        supply_limit = obs.observation.player[4]                         # The supply limit
        army_supply = obs.observation.player[5]                          # The army supply

        # Getting values from the cumulative score system
        killed_unit_score = obs.observation.score_cumulative[5]
        killed_building_score = obs.observation.score_cumulative[6]

        # Defining our state, considering our enemies' positions.
        current_state = np.zeros(20)
        current_state[0] = supply_depot_count
        current_state[1] = barracks_count
        current_state[2] = supply_limit
        current_state[3] = army_supply

        # Insteading of making a vector for all coordnates on the map, we'll discretize our enemy space
        # and use a 16x16 grid to store enemy positions by marking a square as 1 if there's any enemy on it.
        hot_squares = np.zeros(16)
        enemy_y, enemy_x = (obs.observation.feature_minimap[_PLAYER_RELATIVE] == _PLAYER_HOSTILE).nonzero()
        for i in range(0, len(enemy_y)):
            y = int(math.ceil((enemy_y[i] + 1) / 16))
            x = int(math.ceil((enemy_x[i] + 1) / 16))

            hot_squares[((y - 1) * 4) + (x - 1)] = 1

        if not self.base_top_left:
            hot_squares = hot_squares[::-1]

        for i in range(0, 16):
            # Adds 4 to account for supply_depot_count, barracks_count, supply_limit and army_supply
            current_state[i + 4] = hot_squares[i]

        # Calculating the reward by taking the difference between the score system's current and previous values.
        # We only calculate the reward if an action was previously selected.
        if self.previous_action is not None:
            reward = 0

            if killed_unit_score > self.previous_killed_unit_score:
                reward += KILL_UNIT_REWARD

            if killed_building_score > self.previous_killed_building_score:
                reward += KILL_BUILDING_REWARD

            # Learns from the current (s, a, r, s') tuple.
            self.rl_model.learn(str(self.previous_state), self.previous_action, reward, str(current_state))

        ## Handling action selection
        #smart_action = smart_actions[random.randrange(0, len(smart_actions) - 1)]
        rl_action = self.rl_model.choose_action(str(current_state))
        smart_action = smart_actions[rl_action]

        # Saving the score system's current values
        self.previous_killed_unit_score = killed_unit_score
        self.previous_killed_building_score = killed_building_score
        self.previous_state = current_state
        self.previous_action = rl_action

        x = 0
        y = 0

        # Handles the case where we select an attack smart_action by splitting the attack
        # smart action from its coordinates.
        if '_' in smart_action:
            smart_action, x, y = smart_action.split('_')

        ## Binding smart actions to action implementations
        if smart_action == ACTION_DO_NOTHING:
            return actions.FUNCTIONS.no_op()

        elif smart_action == ACTION_SELECT_SCV:
            unit_type = obs.observation.feature_screen[_UNIT_TYPE]
            unit_y, unit_x = (unit_type == _TERRAN_SCV).nonzero()

            if unit_y.any():
                i = random.randint(0, len(unit_y) - 1)
                target = [unit_x[i], unit_y[i]]

                return actions.FUNCTIONS.select_point("select", target)
        
        elif smart_action == ACTION_BUILD_SUPPLY_DEPOT:
            if _BUILD_SUPPLY_DEPOT in obs.observation.available_actions:
                unit_type = obs.observation.feature_screen[_UNIT_TYPE]
                unit_y, unit_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()

                if unit_y.any():
                    target = self.transformDistance(int(unit_x.mean()), 0, int(unit_y.mean()), 20)

                    return actions.FUNCTIONS.Build_SupplyDepot_screen(_NOT_QUEUED, target)
                
        elif smart_action == ACTION_BUILD_BARRACKS:
            if _BUILD_BARRACKS in obs.observation.available_actions:
                unit_type = obs.observation.feature_screen[_UNIT_TYPE]
                unit_y, unit_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()

                if unit_y.any():
                    target = self.transformDistance(int(unit_x.mean()), 20, int(unit_y.mean()), 0)

                    return actions.FUNCTIONS.Build_Barracks_screen(_NOT_QUEUED, target)
        
        elif smart_action == ACTION_SELECT_BARRACKS:
            unit_type = obs.observation.feature_screen[_UNIT_TYPE]
            unit_y, unit_x = (unit_type == _TERRAN_BARRACKS).nonzero()

            if unit_y.any():
                target = [int(unit_x.mean()), int(unit_y.mean())]

                return actions.FUNCTIONS.select_point('select', target)
        
        elif smart_action == ACTION_BUILD_MARINE:
            if _TRAIN_MARINE in obs.observation.available_actions:
                return actions.FUNCTIONS.Train_Marine_quick(_QUEUED)
        
        elif smart_action == ACTION_SELECT_ARMY:
            if _SELECT_ARMY in obs.observation.available_actions:
                return actions.FUNCTIONS.select_army(_NOT_QUEUED)
        
        elif smart_action == ACTION_ATTACK:
            if obs.observation.single_select[0][0] != _TERRAN_SCV and _ATTACK_MINIMAP in obs.observation.available_actions:
                # x, y are strings, so we need to cast them to integers
                return actions.FUNCTIONS.Attack_minimap(_NOT_QUEUED, self.transformLocation(int(x), int(y)))
        
        return actions.FUNCTIONS.no_op()

    
    def transformDistance(self, x, x_distance, y, y_distance):
        if not self.base_top_left:
            return [x - x_distance, y - y_distance]
        
        return [x + x_distance, y + y_distance]

    
    def transformLocation(self, x, y):
        '''Converts absolute x and y values based on the location of our base, instead of just the distance.'''
        if not self.base_top_left:
            return [64 - x, 64 - y]
        
        return [x, y]