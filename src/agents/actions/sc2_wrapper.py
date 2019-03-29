import random
import numpy as np
from .base.abwrapper import ActionWrapper
from agents.actions.sc2 import *
from utils.agent_utils import one_hot_encode_smart_actions, transformDistance, transformLocation
from pysc2.lib import features


## Defining action constants. These are names of the actions our agent will try to use.
ACTION_DO_NOTHING = 'donothing'                 # The agent does nothing for 3 steps
ACTION_BUILD_SUPPLY_DEPOT = 'buildsupplydepot'  # Selects SCV > builds supply depot > send SCV to harvest minerals
ACTION_BUILD_BARRACKS = 'buildbarracks'         # Selects SCV > builds barracks > sends SCV to harvest minerals
ACTION_BUILD_MARINE = 'buildmarine'             # Selects all barracks > trains marines > nothing
ACTION_ATTACK = 'attack'                        # Selects army > attacks coordinates > nothing

_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index

_PLAYER_SELF = 1
_TERRAN_COMMANDCENTER = 18
_TERRAN_SUPPLY_DEPOT = 19
_TERRAN_BARRACKS = 21
_TERRAN_SCV = 45


class SC2Wrapper(ActionWrapper):

    def __init__(self):
        self.move_number = 0

        ## Defining the wrapper's smart actions
        self.smart_actions = [
            ACTION_DO_NOTHING,
            ACTION_BUILD_SUPPLY_DEPOT,
            ACTION_BUILD_BARRACKS,
            ACTION_BUILD_MARINE
        ]

        # Splitting the minimap into a 4x4 grid because the marine's effective range is able to cover
        # the entire map from just this number of cells.
        for mm_x in range(0, 64):
            for mm_y in range(0, 64):
                if (mm_x + 1) % 32 == 0 and (mm_y + 1) % 32 == 0:
                    self.smart_actions.append(ACTION_ATTACK + '_' + str(mm_x - 16) + '_' + str(mm_y - 16))
   
        self.encoded_actions = one_hot_encode_smart_actions(self.smart_actions)
        self.action_space_dim = len(self.get_actions())


    def is_action_done(self):
        return self.move_number == 0

    
    def reset(self):
        self.move_number = 0


    def get_actions(self):
        return self.encoded_actions


    def get_action_space_dim(self):
        return self.action_space_dim

    
    def splitAction(self, smart_action):
        '''Breaks out x, y coordinates from actions if there are any.'''
        x = 0
        y = 0
        if '_' in smart_action:
            smart_action, x, y = smart_action.split('_')

        return (smart_action, x, y)


    def get_excluded_actions(self, obs):
        unit_type = obs.observation.feature_screen[_UNIT_TYPE]

        depot_y, depot_x = (unit_type == _TERRAN_SUPPLY_DEPOT).nonzero()
        supply_depot_count = int(round(len(depot_y) / 69))

        barracks_y, barracks_x = (unit_type == _TERRAN_BARRACKS).nonzero()
        barracks_count = int(round(len(barracks_y) / 137))

        supply_used = obs.observation.player[3]
        supply_limit = obs.observation.player[4]
        supply_free = supply_limit - supply_used

        army_supply = obs.observation.player[5]
        worker_supply = obs.observation.player[6]

        # Adding invalid actions to the list of excluded actions
        excluded_actions = []
        # If the supply depot limit of 2 was reached, removes the ability to build it.
        if supply_depot_count == 2 or worker_supply == 0:
            excluded_actions.append(self.encoded_actions[1])
        # If we have no supply depots or we have 2 barracks, we remove the ability to build it.
        if supply_depot_count == 0 or barracks_count == 2 or worker_supply == 0:
            excluded_actions.append(self.encoded_actions[2])
        # If we don't have any barracks or have reached supply limit, remove the ability to train marines
        if supply_free == 0 or barracks_count == 0:
            excluded_actions.append(self.encoded_actions[3])
        # If we have no marines, we remove attack actions
        if army_supply == 0:
            excluded_actions.append(self.encoded_actions[4])
            excluded_actions.append(self.encoded_actions[5])
            excluded_actions.append(self.encoded_actions[6])
            excluded_actions.append(self.encoded_actions[7])
        
        return excluded_actions


    def get_action(self, one_hot_action, obs):
        smart_action = self.smart_actions[np.argmax(one_hot_action)]
        smart_action, x, y = self.splitAction(smart_action)
        

        # Initializing variables that are used to select the actions
        unit_type = obs.observation.feature_screen[_UNIT_TYPE]
        player_y, player_x = (obs.observation.feature_minimap[_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
        base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0
        player_cc_y, player_cc_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()

        if self.move_number == 0:
            self.move_number += 1

            # Selects a random SCV, this is the first step to building a supply depot or barracks
            if smart_action == ACTION_BUILD_BARRACKS or smart_action == ACTION_BUILD_SUPPLY_DEPOT:
                return select_random_unit(obs)

            # Selects all barracks on the screen simultaneously
            elif smart_action == ACTION_BUILD_MARINE:
                return select_all_barracks(obs)

            elif smart_action == ACTION_ATTACK:
                return select_army(obs)

        elif self.move_number == 1:
            self.move_number += 1

            # Commands the SCV to build the depot at a given location. The place
            # we use to build supply depots and barracks are hard coded.
            if smart_action == ACTION_BUILD_SUPPLY_DEPOT:
                # Calculates the number of supply depots currently built
                unit_type = obs.observation.feature_screen[_UNIT_TYPE]
                depot_y, depot_x = (unit_type == _TERRAN_SUPPLY_DEPOT).nonzero()
                supply_depot_count = int(round(len(depot_y) / 69))

                if supply_depot_count < 2:
                    if player_cc_y.any():
                        # Builds supply depots at a fixed location
                        if supply_depot_count == 0:
                            target = transformDistance(round(player_cc_x.mean()), -35, round(player_cc_y.mean()), 0, base_top_left)
                        elif supply_depot_count == 1:
                            target = transformDistance(round(player_cc_x.mean()), -25, round(player_cc_y.mean()), -25, base_top_left)
                        return build_supply_depot(obs, target) 

            # Commands the selected SCV to build barracks at a given location
            elif smart_action == ACTION_BUILD_BARRACKS:
                # Calculates the number of barracks currently built
                unit_type = obs.observation.feature_screen[_UNIT_TYPE]
                barracks_y, barracks_x = (unit_type == _TERRAN_BARRACKS).nonzero()
                barracks_count = int(round(len(barracks_y) / 137))

                if barracks_count < 2:
                    if player_cc_y.any():
                        # Builds barracks at a fixed location.
                        if barracks_count == 0:
                            target = transformDistance(round(player_cc_x.mean()), 15, round(player_cc_y.mean()), -9, base_top_left)
                        elif barracks_count == 1:
                            target = transformDistance(round(player_cc_x.mean()), 15, round(player_cc_y.mean()), 12, base_top_left)
                        return build_barracks(obs, target)

            # Tells the barracks to train a marine
            elif smart_action == ACTION_BUILD_MARINE:
                return train_marine(obs)
            
            # Tells the agent to attack a location on the map
            elif smart_action == ACTION_ATTACK:
                do_it = True
                
                # Checks if any SCV is selected. If so, the agent doesn't attack.
                if len(obs.observation.single_select) > 0 and obs.observation.single_select[0][0] == _TERRAN_SCV:
                    do_it = False
                if len(obs.observation.multi_select) > 0 and obs.observation.multi_select[0][0] == _TERRAN_SCV:
                    do_it = False
                if do_it:
                    x_offset = random.randint(-1, 1)
                    y_offset = random.randint(-1, 1)
                    target = transformLocation(int(x) + (x_offset * 8), int(y) + (y_offset * 8), base_top_left)
                    return attack_target_point(obs, target)

        elif self.move_number == 2:
            self.move_number = 0

            # Sends the SCV back to a mineral patch after it finished building.
            if smart_action == ACTION_BUILD_BARRACKS or smart_action == ACTION_BUILD_SUPPLY_DEPOT:
                return harvest_point(obs)

        return no_op()

