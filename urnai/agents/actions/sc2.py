import random
import numpy as np
from collections import Counter
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
_BUILD_SENSORTOWER = actions.RAW_FUNCTIONS.Build_SensorTower_pt
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

'''ENGINEERING BAY RESEARCH'''
_RESEARCH_TERRAN_INF_WEAPONS = actions.RAW_FUNCTIONS.Research_TerranInfantryWeapons_quick
_RESEARCH_TERRAN_INF_ARMOR = actions.RAW_FUNCTIONS.Research_TerranInfantryArmor_quick
_RESEARCH_TERRAN_HISEC_AUTOTRACKING = actions.RAW_FUNCTIONS.Research_HiSecAutoTracking_quick
_RESEARCH_TERRAN_NEOSTEEL_FRAME = actions.RAW_FUNCTIONS.Research_NeosteelFrame_quick
_RESEARCH_TERRAN_STRUCTURE_ARMOR = actions.RAW_FUNCTIONS.Research_TerranStructureArmorUpgrade_quick

'''ARMORY RESEARCH'''
_RESEARCH_TERRAN_SHIPS_WEAPONS = actions.RAW_FUNCTIONS.Research_TerranShipWeapons_quick
_RESEARCH_TERRAN_VEHIC_WEAPONS = actions.RAW_FUNCTIONS.Research_TerranVehicleWeapons_quick
_RESEARCH_TERRAN_SHIPVEHIC_PLATES = actions.RAW_FUNCTIONS.Research_TerranVehicleAndShipPlating_quick

'''GHOST ACADEMY RESEARCH'''
_RESEARCH_TERRAN_GHOST_CLOAK = actions.RAW_FUNCTIONS.Research_PersonalCloaking_quick

'''BARRACK RESEARCH'''
_RESEARCH_TERRAN_STIMPACK = actions.RAW_FUNCTIONS.Research_Stimpack_quick
_RESEARCH_TERRAN_COMBATSHIELD = actions.RAW_FUNCTIONS.Research_CombatShield_quick
_RESEARCH_TERRAN_CONCUSSIVESHELL = actions.RAW_FUNCTIONS.Research_ConcussiveShells_quick

'''FACTORY RESEARCH'''
_RESEARCH_TERRAN_INFERNAL_PREIGNITER = actions.RAW_FUNCTIONS.Research_InfernalPreigniter_quick
_RESEARCH_TERRAN_DRILLING_CLAWS = actions.RAW_FUNCTIONS.Research_DrillingClaws_quick
# check if these two following research options are actually from the factory building
_RESEARCH_TERRAN_CYCLONE_LOCKONDMG = actions.RAW_FUNCTIONS.Research_CycloneLockOnDamage_quick
_RESEARCH_TERRAN_CYCLONE_RAPIDFIRE = actions.RAW_FUNCTIONS.Research_CycloneRapidFireLaunchers_quick

'''STARPORT RESEARCH'''
_RESEARCH_TERRAN_HIGHCAPACITYFUEL = actions.RAW_FUNCTIONS.Research_HighCapacityFuelTanks_quick
_RESEARCH_TERRAN_CORVIDREACTOR = actions.RAW_FUNCTIONS.Research_RavenCorvidReactor_quick
_RESEARCH_TERRAN_BANSHEECLOAK = actions.RAW_FUNCTIONS.Research_BansheeCloakingField_quick
_RESEARCH_TERRAN_BANSHEEHYPERFLIGHT = actions.RAW_FUNCTIONS.Research_BansheeHyperflightRotors_quick
_RESEARCH_TERRAN_ADVANCEDBALLISTICS = actions.RAW_FUNCTIONS.Research_AdvancedBallistics_quick

'''FUSION CORE RESEARCH'''
_RESEARCH_TERRAN_BATTLECRUISER_WEAPONREFIT = actions.RAW_FUNCTIONS.Research_BattlecruiserWeaponRefit_quick

'''TRAINING ACTIONS'''
_TRAIN_SCV = actions.RAW_FUNCTIONS.Train_SCV_quick
_TRAIN_MARINE = actions.RAW_FUNCTIONS.Train_Marine_quick
_TRAIN_MARAUDER = actions.RAW_FUNCTIONS.Train_Marauder_quick
_TRAIN_REAPER = actions.RAW_FUNCTIONS.Train_Reaper_quick
_TRAIN_GHOST = actions.RAW_FUNCTIONS.Train_Ghost_quick
_TRAIN_HELLION = actions.RAW_FUNCTIONS.Train_Hellion_quick
_TRAIN_HELLBAT = actions.RAW_FUNCTIONS.Train_Hellbat_quick
_TRAIN_SIEGETANK = actions.RAW_FUNCTIONS.Train_SiegeTank_quick
_TRAIN_CYCLONE = actions.RAW_FUNCTIONS.Train_Cyclone_quick
_TRAIN_WIDOWMINE = actions.RAW_FUNCTIONS.Train_WidowMine_quick
_TRAIN_THOR = actions.RAW_FUNCTIONS.Train_Thor_quick
_TRAIN_VIKING = actions.RAW_FUNCTIONS.Train_VikingFighter_quick
_TRAIN_MEDIVAC = actions.RAW_FUNCTIONS.Train_Medivac_quick
_TRAIN_LIBERATOR = actions.RAW_FUNCTIONS.Train_Liberator_quick
_TRAIN_RAVEN = actions.RAW_FUNCTIONS.Train_Raven_quick
_TRAIN_BANSHEE = actions.RAW_FUNCTIONS.Train_Banshee_quick
_TRAIN_BATTLECRUISER = actions.RAW_FUNCTIONS.Train_Battlecruiser_quick

'''UNIT EFFECTS'''
_EFFECT_STIMPACK = actions.RAW_FUNCTIONS.Effect_Stim_quick


# PROTOSS ACTIONS

_BUILD_PYLON = actions.RAW_FUNCTIONS.Build_Pylon_pt

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

def get_idle_worker(obs, player_race):
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

def get_all_idle_workers(obs, player_race):
    if player_race == _PROTOSS:
        workers = get_my_units_by_type(obs, units.Protoss.Probe)
    elif player_race == _TERRAN:
        workers = get_my_units_by_type(obs, units.Terran.SCV)
    elif player_race == _ZERG:
        workers = get_my_units_by_type(obs, units.Zerg.Drone)

    idle_workers = []

    if len(workers) > 0:
        for worker in workers:
            if worker.order_length == 0: # checking if worker is idle
                idle_workers.append(worker)
        return idle_workers
    return _NO_UNITS

def get_closest_unit(obs, target_xy, unit_type = _NO_UNITS, units_list = _NO_UNITS):
    if unit_type != _NO_UNITS:
        units = get_my_units_by_type(obs, unit_type)
        if len(units) > 0:
            distances = get_distances(obs, units, target_xy)
            min_dist_index = np.argmin(distances)
            unit = units[min_dist_index]
            return unit

    elif units_list != _NO_UNITS:
        if len(units_list) == 0:
            units_list = [units_list]
        distances = get_distances(obs, units_list, target_xy)
        min_dist_index = np.argmin(distances)
        unit = units_list[min_dist_index]
        return unit
    return _NO_UNITS

def build_structure_by_type(obs, action_id, player_race, target=None):
    if player_race == _TERRAN:
        worker = select_random_unit_by_type(obs, units.Terran.SCV)
    elif player_race == _PROTOSS:
        worker = select_random_unit_by_type(obs, units.Protoss.Probe)
    else:
        worker = select_random_unit_by_type(obs, units.Zerg.Drone)
    
    if worker != _NO_UNITS and target != _NO_UNITS:
        if " raw_cmd " in str(action_id.function_type):                 # Checking if the build action is of type RAW_CMD
            return action_id("now", target.tag), _NO_UNITS              # RAW_CMD actions only need a [0]queue and [1]unit_tags and doesn't use a worker (i think)
        
        elif " raw_cmd_pt " in str(action_id.function_type):            # Checking if the build action is of type RAW_CMD_PT
            if is_valid_target(target):                
                return action_id("now", worker.tag, target), worker     # RAW_CMD_PT actions need a [0]queue, [1]unit_tags and [2]world_point

        elif " raw_cmd_unit " in str(action_id.function_type):          # Checking if the build action is of type RAW_CMD_UNIT
            return action_id("now", worker.tag, target.tag), worker     # RAW_CMD_UNIT actions need a [0]queue, [1]unit_tags and [2]unit_tags
    return _NO_OP(), _NO_UNITS


def research_upgrade(obs, action_id, building_type):
    if building_exists(obs, building_type):
        buildings = get_my_units_by_type(obs, building_type)
        for building in buildings:
            if building.build_progress == 100 and building.order_progress_0 == 0:
                return action_id("now", building.tag)
    return _NO_OP()


def effect_units(obs, action_id, units):
    if units != _NO_UNITS:
        unit = units[0]
        units.pop(0)
        if len(units) == 0:
            units = _NO_UNITS
        return action_id("now", unit.tag), units
    return no_op()


def train_unit(obs, action_id, building_type):
    buildings = get_my_units_by_type(obs, building_type)
    if len(buildings) > 0:
        for building in buildings:
            if building.build_progress == 100 and building.order_progress_0 == 0:
                if building.assigned_harvesters <= building.ideal_harvesters:
                    return action_id("now", building.tag)
    return _NO_OP()


def attack_target_point(obs, units, target):
    if units != _NO_UNITS:
        distances = get_distances(obs, units, target)
        unit_index = np.argmax(distances)
        unit = units[unit_index]
        units.pop(unit_index)
        if len(units) == 0:
            units = _NO_UNITS
        return actions.RAW_FUNCTIONS.Attack_pt("now", unit.tag, target), units
    return no_op()

def attack_distribute_army(obs, units):
    if units != _NO_UNITS:
        unit = units[0]
        x_offset = random.randint(-8, 8)
        y_offset = random.randint(-8, 8)
        target = [unit.x + x_offset, unit.y + y_offset]
        units.pop(0)
        if len(units) == 0:
            units = _NO_UNITS
        return actions.RAW_FUNCTIONS.Attack_pt("now", unit.tag, target), units
    return no_op()


def harvest_gather_minerals_quick(obs, worker, player_race):
    if player_race == _TERRAN: 
        townhalls = get_my_units_by_type(obs, units.Terran.CommandCenter)
        townhalls.extend(get_my_units_by_type(obs, units.Terran.PlanetaryFortress))
        townhalls.extend(get_my_units_by_type(obs, units.Terran.OrbitalCommand))
    if player_race == _PROTOSS: townhalls = get_my_units_by_type(obs, units.Protoss.Nexus)
    if player_race == _ZERG: townhalls = get_my_units_by_type(obs, units.Zerg.Hatchery)

    if worker != _NO_UNITS:
        mineral_fields = get_neutral_units_by_type(obs, units.Neutral.MineralField)
        if len(mineral_fields) > 0:
            # Checks every townhall if it is able to receive workers. If it is, searches for the closest mineral field
            # If we find one, send the worker to gather minerals there.
            if len(townhalls) > 0:
                for townhall in townhalls:
                    if townhall.assigned_harvesters < townhall.ideal_harvesters:
                        target = [townhall.x, townhall.y]
                        distances = get_distances(obs, mineral_fields, target)
                        closest_mineral = mineral_fields[np.argmin(distances)]
                        return actions.RAW_FUNCTIONS.Harvest_Gather_unit("queued", worker.tag, closest_mineral.tag)
                        # for mineral_field in mineral_fields:
                        #     target = [mineral_field.x, mineral_field.y]
                        #     distances = get_distances(obs, townhalls, target)
                        #     if distances[np.argmin(distances)] < 10:
                        #         return actions.RAW_FUNCTIONS.Harvest_Gather_unit("queued", worker.tag, mineral_field.tag)

    return _NO_OP()


# def harvest_gather_minerals(obs, worker, command_center, player_race):
#     if player_race == _TERRAN: 
#         townhalls = get_my_units_by_type(obs, units.Terran.CommandCenter)
#         townhalls.extend(get_my_units_by_type(obs, units.Terran.PlanetaryFortress))
#         townhalls.extend(get_my_units_by_type(obs, units.Terran.OrbitalCommand))
#     if player_race == _PROTOSS: townhalls = get_my_units_by_type(obs, units.Protoss.Nexus)
#     if player_race == _ZERG: townhalls = get_my_units_by_type(obs, units.Zerg.Hatchery)

#     if worker != _NO_UNITS:
#         mineral_fields = get_neutral_units_by_type(obs, units.Neutral.MineralField)
#         if len(mineral_fields) > 0:
#             target = [command_center.x, command_center.y]
#             distances = get_distances(obs, mineral_fields, target)
#             idx_argmin = np.argmin(distances)
#             if distances[idx_argmin] < 10:
#                 return actions.RAW_FUNCTIONS.Harvest_Gather_unit("queued", worker.tag, mineral_fields[idx_argmin].tag)

#     return _NO_OP()

def harvest_gather_minerals(obs, player_race, idle_workers=False):
    if player_race == _TERRAN: 
        townhalls = get_my_units_by_type(obs, units.Terran.CommandCenter)
        townhalls.extend(get_my_units_by_type(obs, units.Terran.PlanetaryFortress))
        townhalls.extend(get_my_units_by_type(obs, units.Terran.OrbitalCommand))
    if player_race == _PROTOSS: townhalls = get_my_units_by_type(obs, units.Protoss.Nexus)
    if player_race == _ZERG: townhalls = get_my_units_by_type(obs, units.Zerg.Hatchery)

    mineral_fields = get_neutral_units_by_type(obs, units.Neutral.MineralField)
    if len(mineral_fields) > 0:
        # Checks every townhall if it is able to receive workers. If it is, searches for mineral fields closer than 10 units of distance.
        # If we find one, send the worker to gather minerals there.
        if len(townhalls) > 0:
            for townhall in townhalls:
                if townhall.assigned_harvesters < townhall.ideal_harvesters:
                    target = [townhall.x, townhall.y]
                    if idle_workers:
                        worker = get_closest_unit(obs, target, units_list=idle_workers)
                    else:
                        worker = get_closest_unit(obs, target, unit_type=units.Terran.SCV)
                    if worker.order_id_0 == 362 or worker.order_id_0 == 359 or worker.order_length == 0:
                        for townhall in townhalls:
                            if townhall.assigned_harvesters < townhall.ideal_harvesters:
                                target = [townhall.x, townhall.y]
                                distances = get_distances(obs, mineral_fields, target)
                                closest_mineral = mineral_fields[np.argmin(distances)]
                                return actions.RAW_FUNCTIONS.Harvest_Gather_unit("queued", worker.tag, closest_mineral.tag)

    return _NO_OP()


def harvest_gather_gas(obs, worker, refinery):
    if worker != _NO_UNITS:
        return actions.RAW_FUNCTIONS.Harvest_Gather_unit("queued", worker.tag, refinery.tag)
    return _NO_OP()


def harvest_return(obs, worker):
    if worker != _NO_UNITS:
        return actions.RAW_FUNCTIONS.Harvest_Return_quick("queued", worker.tag)
    return _NO_OP()


def build_structure_raw(obs, building_type, building_action, move_number, last_worker, max_amount = 999):

    player_race = get_unit_race(building_type)

    if get_units_amount(obs, building_type) < max_amount:
        if move_number == 0:
            move_number += 1

            buildings = get_my_units_by_type(obs, building_type)
            if len(buildings) > 0:
                target = random.choice(buildings)
                action, last_worker = build_structure_by_type(obs, building_action, player_race, target)
                return action, last_worker, move_number

        if move_number == 1:
            move_number +=1
            return harvest_gather_minerals_quick(obs, last_worker, player_race), last_worker, move_number
        if move_number == 2:
            move_number = 0
    return _NO_OP(), last_worker, move_number


def build_structure_raw_pt(obs, building_type, building_action, move_number, last_worker, base_top_left, max_amount = 999):
    ybrange=0 if base_top_left else 32
    ytrange=32 if base_top_left else 63

    player_race = get_unit_race(building_type)
        
    if get_units_amount(obs, building_type) < max_amount:
        if move_number == 0:
            move_number += 1
            x = random.randint(0,63)
            y = random.randint(ybrange, ytrange)
            target = [x, y]
            action, last_worker = build_structure_by_type(obs, building_action, player_race, target)
            return action, last_worker, move_number
        if move_number == 1:
            move_number +=1
            return harvest_gather_minerals_quick(obs, last_worker, player_race), last_worker, move_number
        if move_number == 2:
            move_number = 0
    return _NO_OP(), last_worker, move_number


def build_structure_raw_pt2(obs, building_type, building_action, move_number, last_worker, base_top_left, max_amount = 999, targets = []):
    ybrange=0 if base_top_left else 32
    ytrange=32 if base_top_left else 63

    player_race = get_unit_race(building_type)

    building_amount = get_units_amount(obs, building_type)
    if len(targets) == 0 or building_amount >= len(targets):
        target = [random.randint(0,63), random.randint(ybrange, ytrange)]
    else:
        target = targets[building_amount]
        if not base_top_left: target = (63-target[0]-5, 63-target[1]+5)
        
    if building_amount < max_amount:
        if move_number == 0:
            move_number += 1
            action, last_worker = build_structure_by_type(obs, building_action, player_race, target)
            return action, last_worker, move_number
        if move_number == 1:
            move_number +=1
            return harvest_gather_minerals_quick(obs, last_worker, player_race), last_worker, move_number
        if move_number == 2:
            move_number = 0
    return _NO_OP(), last_worker, move_number


def build_gas_structure_raw_unit(obs, building_type, building_action, player_race, move_number, last_worker, max_amount = 999):  

    player_race = get_unit_race(building_type)

    if get_units_amount(obs, building_type) < max_amount:
        if move_number == 0:
            move_number += 1
            chosen_geyser = get_exploitable_geyser(obs, player_race)
            action, last_worker = build_structure_by_type(obs, building_action, player_race, chosen_geyser)
            return action, last_worker, move_number
        if move_number == 1:
            move_number +=1
            return harvest_gather_minerals_quick(obs, last_worker, player_race), last_worker, move_number
        if move_number == 2:
            move_number = 0
    return _NO_OP(), last_worker, move_number

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

# TO DO: Implement the following methods to facilitate checks and overall code reuse:

# Create a "get my units by types" where we pass instead of a single type an array of unit types and the return is an array of those units from the chosen types:
# possible function prototype: get_my_units_by_types(obs, unit_types) (maybe we can just reuse the get_my_units_by_type function and create a verification if unit_type is a single type or array of types)

# check_unit_validity (should check if the object im receiving is a proper unit from pysc2)

def select_army(obs, player_race):
    army = []
    if player_race == _PROTOSS:
        army.extend(get_my_units_by_type(obs, units.Protoss.Zealot))
    elif player_race == _TERRAN:
        army.extend(get_my_units_by_type(obs, units.Terran.Marine))
        army.extend(get_my_units_by_type(obs, units.Terran.Marauder))
        army.extend(get_my_units_by_type(obs, units.Terran.Reaper))
        army.extend(get_my_units_by_type(obs, units.Terran.Ghost))
        army.extend(get_my_units_by_type(obs, units.Terran.Hellion))
        army.extend(get_my_units_by_type(obs, units.Terran.Hellbat))
        army.extend(get_my_units_by_type(obs, units.Terran.SiegeTank))
        army.extend(get_my_units_by_type(obs, units.Terran.Cyclone))
        army.extend(get_my_units_by_type(obs, units.Terran.WidowMine))
        army.extend(get_my_units_by_type(obs, units.Terran.Thor))
        army.extend(get_my_units_by_type(obs, units.Terran.ThorHighImpactMode))
        army.extend(get_my_units_by_type(obs, units.Terran.VikingAssault))
        army.extend(get_my_units_by_type(obs, units.Terran.VikingFighter))
        army.extend(get_my_units_by_type(obs, units.Terran.Medivac))
        army.extend(get_my_units_by_type(obs, units.Terran.Liberator))
        army.extend(get_my_units_by_type(obs, units.Terran.LiberatorAG))
        army.extend(get_my_units_by_type(obs, units.Terran.Raven))
        army.extend(get_my_units_by_type(obs, units.Terran.Banshee))
        army.extend(get_my_units_by_type(obs, units.Terran.Battlecruiser))
    elif player_race == _ZERG:
        army.extend(get_my_units_by_type(obs, units.Zerg.Zergling))
    if len(army) == 0:
        army = _NO_UNITS
    return army

# Reduces a matrix "resolution" by a reduction factor. If we have a 64x64 matrix and rf=4 the map will be reduced to 16x16 in which
# every new element of the matrix is an average from 4x4=16 elements from the original matrix
def lower_featuremap_resolution(map, rf):   #rf = reduction_factor
    if rf == 1: return map
    
    N, M = map.shape
    N = N//rf
    M = M//rf

    reduced_map = np.empty((N, M))
    for i in range(N):
        for j in range(M):
            #reduction_array = map[rf*i:rf*i+rf, rf*j:rf*j+rf].flatten()
            #reduced_map[i,j]  = Counter(reduction_array).most_common(1)[0][0]
            
            reduced_map[i,j] = (map[rf*i:rf*i+rf, rf*j:rf*j+rf].sum())/(rf*rf)

    return reduced_map

def get_unit_race(unit_type):
    if unit_type in units.Terran: 
        return _TERRAN
    if unit_type in units.Protoss:
        return _PROTOSS
    if unit_type in units.Zerg:
        return _ZERG
