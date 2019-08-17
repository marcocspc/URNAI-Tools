import random
from pysc2.lib import actions, features, units

'''
An action set defines all actions an agent can use. In the case of StarCraft 2 using PySC2, some actions require extra
processing to work, so it's up to the developper to come up with a way to make them work.

Even though this is not called action_wrapper, this actually acts as a wrapper

e.g: select_point is a function implemented in PySC2 that requires some extra arguments to work, like which point to select.
Using the action_set we can define a way to select which point select_point will use.
Following this example, we could implement a select_random_unit function which processes a random unit from the observation
and returns the corresponding PySC2 call to select_point that would select this processed unit.
'''

## TODO: Move constants to a separate file, so they can be imported and used by other modules
## Defining constants for action ids, so our agent can check if an action is valid
_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_BUILD_SUPPLY_DEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
_BUILD_REFINERY = actions.FUNCTIONS.Build_Refinery_screen.id
_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id
_HARVEST_GATHER = actions.FUNCTIONS.Harvest_Gather_screen.id
_TRAIN_SCV = actions.FUNCTIONS.Train_SCV_quick.id

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

_NOT_QUEUED = [0]
_QUEUED = [1]
_SELECT_ALL = [2]


def no_op():
    return actions.FUNCTIONS.no_op()


def select_random_scv(obs):
    scvs = get_units_by_type(obs, units.Terran.SCV)

    if len(scvs) > 0:
        random_scv = random.choice(scvs)

        # return actions.FUNCTIONS.select_point(_NOT_QUEUED, (random_scv.x, random_scv.y))
        return select_point(_NOT_QUEUED, (random_scv.x, random_scv.y))
    
    return actions.FUNCTIONS.no_op()


def select_all_barracks(obs):
    barracks = get_units_by_type(obs, units.Terran.Barracks)

    if len(barracks) > 0:
        random_barrack = random.choice(barracks)

        # return actions.FUNCTIONS.select_point(_SELECT_ALL, (random_barrack.x, random_barrack.y))
        return select_point(_SELECT_ALL, (random_barrack.x, random_barrack.y))

    return actions.FUNCTIONS.no_op()


def select_command_center(obs):
    command_centers = get_units_by_type(obs, units.Terran.CommandCenter)

    if len(command_centers) > 0:
        command_center = random.choice(command_centers)
        # return actions.FUNCTIONS.select_point(_NOT_QUEUED, (command_center.x, command_center.y))
        return select_point(_NOT_QUEUED, (command_center.x, command_center.y))
    return actions.FUNCTIONS.no_op()


def select_point(select_type, target):
    if check_target_validity(target):
        return actions.FUNCTIONS.select_point(select_type, target)


def select_army(obs):
    if _SELECT_ARMY in obs.available_actions:
        return actions.FUNCTIONS.select_army(_NOT_QUEUED)
    return actions.FUNCTIONS.no_op()


def build_supply_depot(obs, target):
    if _BUILD_SUPPLY_DEPOT in obs.available_actions:
        if check_target_validity(target):
            return actions.FUNCTIONS.Build_SupplyDepot_screen(_NOT_QUEUED, target)
    return actions.FUNCTIONS.no_op()


def build_barracks(obs, target):
    if _BUILD_BARRACKS in obs.available_actions:
        if check_target_validity(target):
            return actions.FUNCTIONS.Build_Barracks_screen(_NOT_QUEUED, target)
    return actions.FUNCTIONS.no_op()


def build_refinery(obs, target):
    if _BUILD_REFINERY in obs.available_actions:
        if check_target_validity(target):
            return actions.FUNCTIONS.Build_Refinery_screen(_NOT_QUEUED, target)
    return actions.FUNCTIONS.no_op()


def train_marine(obs):
    if _TRAIN_MARINE in obs.available_actions:
        return actions.FUNCTIONS.Train_Marine_quick(_QUEUED)
    return actions.FUNCTIONS.no_op()


def train_scv(obs):
    if _TRAIN_SCV in obs.available_actions:
        return actions.FUNCTIONS.Train_SCV_quick(_QUEUED)
    return actions.FUNCTIONS.no_op()


def attack_target_point(obs, target):
    if _ATTACK_MINIMAP in obs.available_actions:
        if check_target_validity(target):
            return actions.FUNCTIONS.Attack_minimap(_NOT_QUEUED, target)
    return actions.FUNCTIONS.no_op()


def harvest_point(obs):
    if _HARVEST_GATHER in obs.available_actions:
        mineral_fields = get_units_by_type(obs, units.Neutral.MineralField)

        if len(mineral_fields) > 0:
            mineral_field = random.choice(mineral_fields)

            if check_target_validity((mineral_field.x, mineral_field.y)):
                return actions.FUNCTIONS.Harvest_Gather_screen(_QUEUED, (mineral_field.x, mineral_field.y))

    return actions.FUNCTIONS.no_op()


#The following methods are used to aid in various mechanical operations the agent has to perform,
#such as: getting all units from a certain type, counting the amount of free supply, etc

def get_units_by_type(obs, unit_type):
    return [unit for unit in obs.feature_units
            if unit.unit_type == unit_type]

def get_free_supply(obs):
    return obs.player[4] - obs.player[3]

def get_units_amount(obs, unit_type):
    return len(get_units_by_type(obs, unit_type))

# TO DO: Find a more general way of checking whether the target is at a valid screen point
def check_target_validity(target):
    if 0 <= target[0] <= 83 and 0<= target[1] <= 83:
        return True
    return False

# TO DO: Implement these methods to facilitate checks and overall code reuse
# valid_target() (checks if the targeted place is valid)
# already_pending()
# can_afford()
# building_exists()
# redistribute_workers()

