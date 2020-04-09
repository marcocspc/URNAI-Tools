import random
import numpy as np
from collections import Counter
from pysc2.lib import actions, features, units
from pysc2.env import sc2_env

'''
An action set defines all actions an agent can use. In the case of StarCraft 2 using PySC2, some actions require extra
processing to work, so it's up to the developper to come up with a way to make them work.

Even though this is not called an action_wrapper, it effectively acts as a wrapper

e.g: actions.RAW_FUNCTIONS.Build_Barracks_pt is a function implemented in PySC2 that requires some extra arguments to work, like 
whether to build it now or to queue the action, which worker is going to perform this action, and the target (given by a [x, y] position)

In this file we sort all of this issues, like deciding when to do an action, which units to use for it, and where to build.
The methods in here effectively serve as a bridge between our high level actions defined in sc2_wrapper.py and the PySC2 library.

'''

## Defining constants for action ids, so our agent can check if an action is valid, and pass these actions as arguments to functions easily
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

def build_structure_by_type(obs, action_id, player_race, target=None):
    if player_race == _TERRAN:
        worker = select_random_unit_by_type(obs, units.Terran.SCV)
    elif player_race == _PROTOSS:
        worker = select_random_unit_by_type(obs, units.Protoss.Probe)
    else:
        worker = select_random_unit_by_type(obs, units.Zerg.Drone)
    
    if worker != _NO_UNITS and target != _NO_UNITS:
        if " raw_cmd " in str(action_id.function_type):                 # Checking if the build action is of type RAW_CMD
            return action_id("now", target.tag), _NO_UNITS              # RAW_CMD actions only need [0]queue and [1]unit_tags and doesn't use a worker
        
        elif " raw_cmd_pt " in str(action_id.function_type):            # Checking if the build action is of type RAW_CMD_PT
            if is_valid_target(target):                
                return action_id("now", worker.tag, target), worker     # RAW_CMD_PT actions need [0]queue, [1]unit_tags and [2]world_point

        elif " raw_cmd_unit " in str(action_id.function_type):          # Checking if the build action is of type RAW_CMD_UNIT
            return action_id("now", worker.tag, target.tag), worker     # RAW_CMD_UNIT actions need [0]queue, [1]unit_tags and [2]unit_tags
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
    return _NO_OP()

def train_unit(obs, action_id, building_type):
    buildings = get_my_units_by_type(obs, building_type)
    if len(buildings) > 0:
        for building in buildings:
            if building.build_progress == 100 and building.order_progress_0 == 0:
                if building.assigned_harvesters <= building.ideal_harvesters:
                    return action_id("now", building.tag)
    return _NO_OP()

# def attack_target_point(obs, units, target):
#     if units != _NO_UNITS:
#         distances = get_distances(obs, units, target)
#         unit_index = np.argmax(distances)
#         unit = units[unit_index]
#         units.pop(unit_index)
#         if len(units) == 0:
#             units = _NO_UNITS
#         return actions.RAW_FUNCTIONS.Attack_pt("now", unit.tag, target), units
#     return no_op()
def attack_target_point(obs, player_race, target, base_top_left):
    if not base_top_left: target = (63-target[0]-5, 63-target[1]+5)
    army = select_army(obs, player_race)
    if army != _NO_UNITS:
        if len(army) > 0:
            distances = list(get_distances(obs, army, target))
        actions_queue = []
        while len(army) != 0:
            unit_index = np.argmax(distances)
            actions_queue.append(actions.RAW_FUNCTIONS.Attack_pt("now", army[unit_index].tag, target))
            army.pop(unit_index)
            distances.pop(unit_index)
        return actions_queue
    return [_NO_OP()]

def attack_distribute_army(obs, player_race):
    army = select_army(obs, player_race)
    if army != _NO_UNITS:
        actions_queue = []
        while len(army) != 0:
            x_offset = random.randint(-8, 8)
            y_offset = random.randint(-8, 8)
            target = [army[0].x + x_offset, army[0].y + y_offset]
            actions_queue.append(actions.RAW_FUNCTIONS.Attack_pt("now", army[0].tag, target))
            army.pop(0)
        return actions_queue
    return [_NO_OP()]

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
                    if townhall.assigned_harvesters <= townhall.ideal_harvesters and townhall.build_progress == 100:
                        target = [townhall.x, townhall.y]
                        closest_mineral = get_closest_unit(obs, target, units_list=mineral_fields)
                        if closest_mineral != _NO_UNITS:
                            return actions.RAW_FUNCTIONS.Harvest_Gather_unit("queued", worker.tag, closest_mineral.tag)

    return _NO_OP()

def harvest_gather_minerals(obs, player_race):
    if player_race == _TERRAN: 
        townhalls = get_my_units_by_type(obs, units.Terran.CommandCenter)
        townhalls.extend(get_my_units_by_type(obs, units.Terran.PlanetaryFortress))
        townhalls.extend(get_my_units_by_type(obs, units.Terran.OrbitalCommand))
        workers = get_my_units_by_type(obs, units.Terran.SCV)
    if player_race == _PROTOSS: 
        townhalls = get_my_units_by_type(obs, units.Protoss.Nexus)
        workers = get_my_units_by_type(obs, units.Protoss.Probe)
    if player_race == _ZERG: 
        townhalls = get_my_units_by_type(obs, units.Zerg.Hatchery)
        workers = get_my_units_by_type(obs, units.Zerg.Drone)

    mineral_fields = get_neutral_units_by_type(obs, units.Neutral.MineralField)
    if len(mineral_fields) > 0:
        # Checks every townhall if it is able to receive workers. If it is, searches for closest mineral field
        # If we find one, send the worker to gather minerals there.
        if len(townhalls) > 0:
            for townhall in townhalls:
                if townhall.assigned_harvesters <= townhall.ideal_harvesters and townhall.build_progress == 100:
                    target = [townhall.x, townhall.y]
                    if len(workers) > 0:
                        distances = list(get_distances(obs, workers, target))
                    while len(workers) != 0:
                        index = np.argmin(distances)
                        if (workers[index].order_id_0 == 362 or workers[index].order_length == 0) and distances[index] >= 2:
                            closest_mineral = get_closest_unit(obs, target, units_list=mineral_fields)
                            if closest_mineral != _NO_UNITS:
                                return actions.RAW_FUNCTIONS.Harvest_Gather_unit("queued", workers[index].tag, closest_mineral.tag)
                        else:
                            workers.pop(index)
                            distances.pop(index)
    return _NO_OP()

def harvest_gather_minerals_idle(obs, player_race, idle_workers):
    if player_race == _TERRAN: 
        townhalls = get_my_units_by_type(obs, units.Terran.CommandCenter)
        townhalls.extend(get_my_units_by_type(obs, units.Terran.PlanetaryFortress))
        townhalls.extend(get_my_units_by_type(obs, units.Terran.OrbitalCommand))
    if player_race == _PROTOSS: 
        townhalls = get_my_units_by_type(obs, units.Protoss.Nexus)
    if player_race == _ZERG: 
        townhalls = get_my_units_by_type(obs, units.Zerg.Hatchery)
    
    mineral_fields = get_neutral_units_by_type(obs, units.Neutral.MineralField)
    if len(mineral_fields) > 0:
        if len(townhalls) > 0:
            for townhall in townhalls:
                if townhall.assigned_harvesters <= townhall.ideal_harvesters and townhall.build_progress == 100:
                    target = [townhall.x, townhall.y]
                    worker = get_closest_unit(obs, target, units_list=idle_workers)
                    if worker != _NO_UNITS:
                        distances = get_distances(obs, mineral_fields, target)
                        closest_mineral_to_townhall = mineral_fields[np.argmin(distances)]
                        return actions.RAW_FUNCTIONS.Harvest_Gather_unit("now", worker.tag, closest_mineral_to_townhall.tag)
    return _NO_OP()

def harvest_gather_gas(obs, player_race):
    if player_race == _TERRAN: 
        gas_colectors = get_my_units_by_type(obs, units.Terran.Refinery)
        workers = get_my_units_by_type(obs, units.Terran.SCV)
    if player_race == _PROTOSS: 
        gas_colectors = get_my_units_by_type(obs, units.Protoss.Assimilator)
        workers = get_my_units_by_type(obs, units.Protoss.Probe) 
    if player_race == _ZERG: 
        gas_colectors = get_my_units_by_type(obs, units.Zerg.Extractor)
        workers = get_my_units_by_type(obs, units.Zerg.Drone)

    if len(gas_colectors) > 0:
        for gas_colector in gas_colectors:
            if 0 <= gas_colector.assigned_harvesters < 3 and gas_colector.build_progress == 100:
                target = [gas_colector.x, gas_colector.y]
                if len(workers) > 0:
                    distances = list(get_distances(obs, workers, target))
                while len(workers) != 0:
                    index = np.argmin(distances)
                    if (workers[index].order_id_0 == 362 or workers[index].order_length == 0) and distances[index] >= 3:
                        return actions.RAW_FUNCTIONS.Harvest_Gather_unit("queued", workers[index].tag, gas_colector.tag)
                    else:
                        workers.pop(index)
                        distances.pop(index)
    return _NO_OP()

def harvest_return(obs, worker):
    if worker != _NO_UNITS:
        return actions.RAW_FUNCTIONS.Harvest_Return_quick("queued", worker.tag)
    return _NO_OP()

def build_structure_raw(obs, building_type, building_action, max_amount = 999):

    player_race = get_unit_race(building_type)

    if get_units_amount(obs, building_type) < max_amount:
        buildings = get_my_units_by_type(obs, building_type)
        if len(buildings) > 0:
            target = random.choice(buildings)
            action_one, last_worker = build_structure_by_type(obs, building_action, player_race, target)
            action_two = harvest_gather_minerals_quick(obs, last_worker, player_race)
            actions_queue = [action_one, action_two]
            return actions_queue

    return [_NO_OP()]

def build_structure_raw_pt(obs, building_type, building_action, base_top_left, max_amount = 999, targets = []):
    ybrange=0 if base_top_left else 32
    ytrange=32 if base_top_left else 63

    player_race = get_unit_race(building_type)

    building_amount = get_units_amount(obs, building_type)
    if len(targets) == 0 or building_amount >= len(targets):
        target = [random.randint(0,63), random.randint(ybrange, ytrange)]
    else:
        target = targets[building_amount]
        if not base_top_left: target = (63-target[0]-6, 63-target[1]+6)

    if building_amount < max_amount:
        action_one, last_worker = build_structure_by_type(obs, building_action, player_race, target)
        action_two = harvest_gather_minerals_quick(obs, last_worker, player_race)
        actions_queue = [action_one, action_two]
        return actions_queue
        
    return [_NO_OP()]

def build_gas_structure_raw_unit(obs, building_type, building_action, player_race, max_amount = 999):  
    player_race = get_unit_race(building_type)
    if get_units_amount(obs, building_type) < max_amount:
        chosen_geyser = get_exploitable_geyser(obs, player_race)
        action_one, last_worker = build_structure_by_type(obs, building_action, player_race, chosen_geyser)
        action_two = harvest_gather_minerals_quick(obs, last_worker, player_race)
        actions_queue = [action_one, action_two]
        return actions_queue
    return [_NO_OP()]

'''
The following methods are used to aid in various mechanical operations the agent has to perform,
such as: getting all units from a certain type, counting the amount of free supply, etc
'''

def select_random_unit_by_type(obs, unit_type):
    units = get_my_units_by_type(obs, unit_type)

    if len(units) > 0:
        random_unit = random.choice(units)
        return random_unit
    return _NO_UNITS

def get_random_idle_worker(obs, player_race):
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
        if len(units_list) != 0:
            distances = get_distances(obs, units_list, target_xy)
            min_dist_index = np.argmin(distances)
            unit = units_list[min_dist_index]
            return unit
    return _NO_UNITS

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
    if len(units) > 0:
        units_xy = [(unit.x, unit.y) for unit in units]
        return np.linalg.norm(np.array(units_xy) - np.array(xy), axis=1)
    pass

def get_euclidean_distance(unit_xy, xy):
    return np.linalg.norm(np.array(unit_xy) - np.array(xy))

def organize_queue(actions, actions_queue):
    action = actions.pop(0)
    while len(actions) > 0:
        actions_queue.append(actions.pop(0))
    return action, actions_queue

# TO DO: Implement the following methods to facilitate checks and overall code reuse:

# Create a "get my units by types" where we pass instead of a single type an array of unit types and the return is an array of those units from the chosen types:
# possible function prototype: get_my_units_by_types(obs, unit_types) (maybe we can just reuse the get_my_units_by_type function and create a verification if unit_type is a single type or array of types)

# check_unit_validity (should check if the object being received is a proper unit from pysc2)

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
    

def get_unit_race(unit_type):
    if unit_type in units.Terran: 
        return _TERRAN
    if unit_type in units.Protoss:
        return _PROTOSS
    if unit_type in units.Zerg:
        return _ZERG

"""
move a unit to a new position
based on https://gist.github.com/fyr91/168996a23f5675536dbf6f1cf75b30d6#file-defeat_zerglings_banelings_env_5-py-L41
"""
def move_to(obs, unit, dest_x, dest_y):
    target = [dest_x, dest_y]
    try:
        return actions.RAW_FUNCTIONS.Move_pt("now", unit.tag, target)
    except:
        return no_op()

