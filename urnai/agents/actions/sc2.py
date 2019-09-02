import random
from pysc2.lib import actions, features, units
from pysc2.env import sc2_env

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
_NO_OP = actions.RAW_FUNCTIONS.no_op

_HARVEST_GATHER = actions.FUNCTIONS.Harvest_Gather_screen.id

_BUILD_SUPPLY_DEPOT = actions.RAW_FUNCTIONS.Build_SupplyDepot_pt
_BUILD_BARRACKS = actions.RAW_FUNCTIONS.Build_Barracks_pt
_BUILD_REFINERY = actions.RAW_FUNCTIONS.Build_Refinery_pt

_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id


_TRAIN_SCV = actions.RAW_FUNCTIONS.Train_SCV_quick
_TRAIN_MARINE = actions.RAW_FUNCTIONS.Train_Marine_quick


_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_PLAYER_ID = features.SCREEN_FEATURES.player_id.index

_PLAYER_SELF = 1
_PLAYER_HOSTILE = 4
_ARMY_SUPPLY = 5

_NOT_QUEUED = [0]
_QUEUED = [1]
_SELECT_ALL = [2]

_NO_UNITS = "no_units"
_TERRAN = sc2_env.Race.terran
_PROTOSS = sc2_env.Race.protoss
_ZERG = sc2_env.Race.zerg


def no_op():
    return actions.FUNCTIONS.no_op()


def select_random_unit_by_type(obs, unit_type):
    units = get_my_units_by_type(obs, unit_type)

    if len(units) > 0:
        random_unit = random.choice(units)
        return random_unit
    return _NO_UNITS

def select_idle_worker(obs, player_race):
    if player_race == _TERRAN:
        workers = get_my_units_by_type(obs, units.Terran.SCV)
    elif player_race == _PROTOSS:
        workers = get_my_units_by_type(obs, units.Protoss.Probe)
    elif player_race == _ZERG:
        workers = get_my_units_by_type(obs, units.Zerg.Drone)

    if len(workers) > 0:
        for worker in workers:
            if worker.order_length == 0: # check if worker is idle
                return worker
    return _NO_UNITS

# TO DO: Implement a select_closest_unit_by_type (useful to select workers closest to building target)

# Convert to raw obs
def select_point(select_type, target):
    if check_target_validity(target):
        return actions.FUNCTIONS.select_point(select_type, target)
    return no_op()

# Convert to raw obs
def select_army(obs):
    return actions.FUNCTIONS.select_army(_NOT_QUEUED)


def build_structure_by_type(obs, action_id, target):
    worker = select_random_unit_by_type(obs, units.Terran.SCV)
    if worker != _NO_UNITS:
        # worker_tag = worker.tag
        if check_target_validity(target):
            return action_id("now", worker.tag, target), worker
    return no_op()

# Convert to raw obs
def train_marine(obs):
    if _TRAIN_MARINE in obs.available_actions:
        return actions.FUNCTIONS.Train_Marine_quick(_QUEUED)
    return no_op()

# TO DO: Check if we can train units using list of structures (get_my_units_by_type)
# or if we need to specify a single structure
def train_scv(obs):
    command_centers = get_my_units_by_type(obs, units.Terran.CommandCenter)
    if len(command_centers) > 0:
        return actions.RAW_FUNCTIONS.Train_SCV_quick("now", command_centers)
    return no_op()

# Convert to raw obs
def attack_target_point(obs, target):
    if _ATTACK_MINIMAP in obs.available_actions:
        if check_target_validity(target):
            return actions.FUNCTIONS.Attack_minimap(_NOT_QUEUED, target)
    return no_op()

# Convert to raw obs
def harvest_point(obs, worker):
    if worker != _NO_UNITS:
        mineral_fields = get_neutral_units_by_type(obs, units.Neutral.MineralField)

        if len(mineral_fields) > 0:
            mineral_field = random.choice(mineral_fields)
            target = [mineral_field.x, mineral_field.y]
            return actions.RAW_FUNCTIONS.Harvest_Gather_SCV_pt("queued", mineral_field.tag, target)
    return no_op()


#The following methods are used to aid in various mechanical operations the agent has to perform,
#such as: getting all units from a certain type, counting the amount of free supply, etc

def get_my_units_by_type(obs, unit_type):
    return [unit for unit in obs.raw_units 
            if unit.unit_type == unit_type
            and unit.alliance == features.PlayerRelative.SELF]

def get_neutral_units_by_type(obs, unit_type):
    return [unit for unit in obs.raw_units 
            if unit.unit_type == unit_type
            and unit.alliance == features.PlayerRelative.NEUTRAL]

def get_free_supply(obs):
    return obs.player.food_cap - obs.player.food_used

def get_units_amount(obs, unit_type):
    return len(get_my_units_by_type(obs, unit_type))

# TO DO: Find a more general way of checking whether the target is at a valid screen point
def check_target_validity(target):
    if 0 <= target[0] <= 63 and 0<= target[1] <= 63:
        return True
    return False

def building_exists(obs, unit_type):
    if get_units_amount(obs, unit_type) > 0:
        return True
    return False

# TO DO: Implement these methods to facilitate checks and overall code reuse
# already_pending()
# can_afford()
# building_exists()
# redistribute_workers()

