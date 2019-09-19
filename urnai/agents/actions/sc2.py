import random
import numpy as np
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
_NO_OP = actions.FUNCTIONS.no_op

_BUILD_COMMAND_CENTER = actions.RAW_FUNCTIONS.Build_CommandCenter_pt
_BUILD_SUPPLY_DEPOT = actions.RAW_FUNCTIONS.Build_SupplyDepot_pt
_BUILD_REFINERY = actions.RAW_FUNCTIONS.Build_Refinery_pt
_BUILD_ENGINEERINGBAY = actions.RAW_FUNCTIONS.Build_EngineeringBay_pt
_BUILD_ARMORY = actions.RAW_FUNCTIONS.Build_Armory_pt
_BUILD_MISSILETURRET = actions.RAW_FUNCTIONS.Build_MissileTurret_pt
_BUILD_BUNKER = actions.RAW_FUNCTIONS.Build_Bunker_pt
_BUILD_FUSIONCORE = actions.RAW_FUNCTIONS.Build_FusionCore_pt
_BUILD_GHOSTACADEMY = actions.RAW_FUNCTIONS.Build_GhostAcademy_pt
_BUILD_BARRACKS = actions.RAW_FUNCTIONS.Build_Barracks_pt
_BUILD_FACTORY = actions.RAW_FUNCTIONS.Build_Factory_pt
_BUILD_STARPORT = actions.RAW_FUNCTIONS.Build_Starport_pt
_BUILD_TECHLAB_BARRACKS = actions.RAW_FUNCTIONS.Build_TechLab_Barracks_quick
_BUILD_TECHLAB_FACTORY = actions.RAW_FUNCTIONS.Build_TechLab_Factory_quick
_BUILD_TECHLAB_STARPORT = actions.RAW_FUNCTIONS.Build_TechLab_Starport_quick
_BUILD_REACTOR_BARRACKS = actions.RAW_FUNCTIONS.Build_Reactor_Barracks_quick
_BUILD_REACTOR_FACTORY = actions.RAW_FUNCTIONS.Build_Reactor_Factory_quick
_BUILD_REACTOR_STARPORT = actions.RAW_FUNCTIONS.Build_Reactor_Starport_quick

_RESEARCH_TERRAN_INF_WEAPONS = actions.RAW_FUNCTIONS.Research_TerranInfantryWeapons_quick
_RESEARCH_TERRAN_INF_ARMOR = actions.RAW_FUNCTIONS.Research_TerranInfantryArmor_quick
_RESEARCH_TERRAN_SHIPS_WEAPONS = actions.RAW_FUNCTIONS.Research_TerranShipWeapons_quick
_RESEARCH_TERRAN_VEHIC_WEAPONS = actions.RAW_FUNCTIONS.Research_TerranVehicleWeapons_quick
_RESEARCH_TERRAN_SHIPVEHIC_PLATES = actions.RAW_FUNCTIONS.Research_TerranVehicleAndShipPlating_quick
_RESEARCH_TERRAN_STRUCTURE_ARMOR = actions.RAW_FUNCTIONS.Research_TerranStructureArmorUpgrade_quick

'''BARRACK RESEARCH'''
_RESEARCH_TERRAN_STIMPACK = actions.RAW_FUNCTIONS.Research_Stimpack_quick
_RESEARCH_TERRAN_COMBATSHIELD = actions.RAW_FUNCTIONS.Research_CombatShield_quick
_RESEARCH_TERRAN_CONCUSSIVESHELL = actions.RAW_FUNCTIONS.Research_ConcussiveShells_quick

'''FACTORY RESEARCH'''
_RESEARCH_TERRAN_INFERNALPREIGNITER = actions.RAW_FUNCTIONS.Research_InfernalPreigniter_quick
_RESEARCH_TERRAN_DRILLING_CLAWS = actions.RAW_FUNCTIONS.Research_DrillingClaws_quick
# check if these two following research options are actually from the factory building
_RESEARCH_TERRAN_CYCLONE_LOCKONDMG = actions.RAW_FUNCTIONS.Research_CycloneLockOnDamage_quick
_RESEARCH_TERRAN_CYCLONE_RAPIDFIRE = actions.RAW_FUNCTIONS.Research_CycloneRapidFireLaunchers_quick

'''STARPORT RESEARCH'''
_RESEARCH_TERRAN_HIGHCAPACITYFUEL = actions.RAW_FUNCTIONS.Research_HighCapacityFuelTanks_quick
_RESEARCH_TERRAN_CORVIDREACTOR = actions.RAW_FUNCTIONS.Research_RavenCorvidReactor_quick
_RESEARCH_TERRAN_BANSHEECLOACK = actions.RAW_FUNCTIONS.Research_BansheeCloakingField_quick
_RESEARCH_TERRAN_BANSHEEHYPERFLIGHT = actions.RAW_FUNCTIONS.Research_BansheeHyperflightRotors_quick
_RESEARCH_TERRAN_ADVANCEDBALLISTICS = actions.RAW_FUNCTIONS.Research_AdvancedBallistics_quick

_TRAIN_SCV = actions.RAW_FUNCTIONS.Train_SCV_quick
_TRAIN_MARINE = actions.RAW_FUNCTIONS.Train_Marine_quick
_TRAIN_MARAUDER = actions.RAW_FUNCTIONS.Train_Marauder_quick

'''Unit Effects'''
_EFFECT_STIMPACK = actions.RAW_FUNCTIONS.Effect_Stim_quick


'''CONSTANTS USED TO DO GENERAL CHECKS'''
_NO_UNITS = "no_units"
_TERRAN = sc2_env.Race.terran
_PROTOSS = sc2_env.Race.protoss
_ZERG = sc2_env.Race.zerg


def no_op():
    return actions.RAW_FUNCTIONS.no_op()


def select_random_unit_by_type(obs, unit_type):
    units = get_my_units_by_type(obs, unit_type)

    if len(units) > 0:
        random_unit = random.choice(units)
        return random_unit
    return _NO_UNITS

def select_idle_worker(obs, player_race):
    if player_race == _PROTOSS:
        workers = get_my_units_by_type(obs, units.Protoss.Probe)
    elif player_race == _TERRAN:
        workers = get_my_units_by_type(obs, units.Terran.SCV)
    elif player_race == _ZERG:
        workers = get_my_units_by_type(obs, units.Zerg.Drone)

    if len(workers) > 0:
        for worker in workers:
            if worker.order_length == 0: # checking if worker is idle
                return worker
    return _NO_UNITS

# TO DO: Implement a select_closest_unit_by_type (useful to select workers closest to building target)

# Convert to raw obs
def select_army(obs):
    return actions.FUNCTIONS.select_army("now")


def build_structure_by_type(obs, action_id, target=None):
    worker = select_random_unit_by_type(obs, units.Terran.SCV)
    if worker != _NO_UNITS and target != _NO_UNITS:
        if " raw_cmd " in str(action_id.function_type):                 # Checking if the build action is of type RAW_CMD
            return action_id("now", target.tag), _NO_UNITS              # RAW_CMD actions only need a [0]queue and [1]unit_tags and doesn't use a worker (i think)
        
        elif " raw_cmd_pt " in str(action_id.function_type):            # Checking if the build action is of type RAW_CMD_PT
            if is_valid_target(target):                
                return action_id("now", worker.tag, target), worker     # RAW_CMD_PT actions need a [0]queue, [1]unit_tags and [2]world_point

        elif " raw_cmd_unit " in str(action_id.function_type):          # Checking if the build action is of type RAW_CMD_UNIT
            return action_id("now", worker.tag, target.tag), worker     # RAW_CMD_UNIT actions need a [0]queue, [1]unit_tags and [2]unit_tags
    return _NO_OP(), _NO_UNITS


def research_upgrade(obs, action_id, target):
    if target != _NO_UNITS:
        return action_id("now", target.tag)
    return _NO_OP()


def effect_units(obs, action_id, units):
    if units != _NO_UNITS:
        for unit in units:
            return action_id("now", unit.tag)
    return _NO_OP()


def train_unit(obs, action_id, building_type):
    buildings = get_my_units_by_type(obs, building_type)
    if len(buildings) > 0:
        for building in buildings:
            if building.build_progress == 100 and building.order_progress_0 == 0:
                return action_id("now", building.tag)
    return _NO_OP()


def attack_target_point(obs, units, target):
    if units != _NO_UNITS:
        for unit in units:
            return actions.RAW_FUNCTIONS.Attack_pt("now", unit.tag, target)
    return no_op()


def harvest_gather_minerals_quick(obs, worker):
    if worker != _NO_UNITS:
        mineral_fields = get_neutral_units_by_type(obs, units.Neutral.MineralField)
        if len(mineral_fields) > 0:
            command_centers = get_my_units_by_type(obs, units.Terran.CommandCenter)
            # Checks for every mineral field if it is closer than 10 units of distance from a command center, if so, sends our worker there to harvest
            if len(command_centers) > 0:
                for mineral_field in mineral_fields:
                    target = [mineral_field.x, mineral_field.y]
                    distances = get_distances(obs, command_centers, target)
                    if distances[np.argmin(distances)] < 10:
                        return actions.RAW_FUNCTIONS.Harvest_Gather_unit("queued", worker.tag, mineral_field.tag)

    return _NO_OP()


def harvest_gather_minerals(obs, worker, command_center):
    if worker != _NO_UNITS:
        mineral_fields = get_neutral_units_by_type(obs, units.Neutral.MineralField)
        if len(mineral_fields) > 0:
            target = [command_center.x, command_center.y]
            distances = get_distances(obs, mineral_fields, target)
            idx_argmin = np.argmin(distances)
            if distances[idx_argmin] < 10:
                return actions.RAW_FUNCTIONS.Harvest_Gather_unit("queued", worker.tag, mineral_fields[idx_argmin].tag)

    return _NO_OP()


def harvest_gather_gas(obs, worker, refinery):
    if worker != _NO_UNITS:
        return actions.RAW_FUNCTIONS.Harvest_Gather_unit("queued", worker.tag, refinery.tag)
    return _NO_OP()

def harvest_return(obs, worker):
    if worker != _NO_UNITS:
        return actions.RAW_FUNCTIONS.Harvest_Return_quick("queued", worker.tag)
    return _NO_OP()


'''
The following methods are used to aid in various mechanical operations the agent has to perform,
such as: getting all units from a certain type, counting the amount of free supply, etc
'''

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
def is_valid_target(target):
    if 0 <= target[0] <= 63 and 0<= target[1] <= 63:
        return True
    return False

def building_exists(obs, unit_type):
    if get_units_amount(obs, unit_type) > 0:
        return True
    return False

def get_exploitable_geyser(obs, player_race):
    if player_race == _PROTOSS:
        townhalls = get_my_units_by_type(obs, units.Protoss.Nexus)
    elif player_race == _TERRAN:
        townhalls = get_my_units_by_type(obs, units.Terran.CommandCenter)
        townhalls.extend(get_my_units_by_type(obs, units.Terran.OrbitalCommand))
        townhalls.extend(get_my_units_by_type(obs, units.Terran.PlanetaryFortress))
    elif player_race == _ZERG:
        townhalls = get_my_units_by_type(obs, units.Zerg.Hatchery)
        townhalls.extend(get_my_units_by_type(obs, units.Zerg.Lair))
        townhalls.extend(get_my_units_by_type(obs, units.Zerg.Hive))
    geysers = get_neutral_units_by_type(obs, units.Neutral.VespeneGeyser)
    if len(geysers) > 0 and len(townhalls) > 0:
        for geyser in geysers:
            for townhall in townhalls:
                if get_euclidean_distance([geyser.x, geyser.y], [townhall.x, townhall.y]) < 10:
                    return geyser
    return _NO_UNITS

def get_distances(obs, units, xy):
    units_xy = [(unit.x, unit.y) for unit in units]
    return np.linalg.norm(np.array(units_xy) - np.array(xy), axis=1)

def get_euclidean_distance(unit_xy, xy):
    return np.linalg.norm(np.array(unit_xy) - np.array(xy))

# TO DO: Implement these methods to facilitate checks and overall code reuse
# check_unit_validity (should check if the object im receiving is a proper unit from pysc2)
# already_pending()
# can_afford()
# building_exists()
# redistribute_workers()

