import random
import numpy as np
from .base.abwrapper import ActionWrapper
from agents.actions.sc2 import *
import agents.actions.sc2 as sc2     # importing our action set file so that we can use its constants

from utils.agent_utils import one_hot_encode, transformDistance, transformLocation
from pysc2.lib import features, units
from pysc2.env import sc2_env


## Defining action constants. These are names of the actions our agent will try to use.
## These are used merely to facilitate checking which actions are being called during code debugging
ACTION_DO_NOTHING = 'donothing'                             # The agent does nothing

ACTION_BUILD_COMMAND_CENTER = 'buildcommandcenter'
ACTION_BUILD_SUPPLY_DEPOT = 'buildsupplydepot'              # Selects SCV > builds supply depot > sends SCV to harvest minerals
ACTION_BUILD_REFINERY = 'buildrefinery'                     # Selects SCV > finds closest vespene geyser and builds a refinery > sends SCV to harvest minerals
ACTION_BUILD_ENGINEERINGBAY = 'buildengineeringbay'
ACTION_BUILD_ARMORY = 'buildarmory'
ACTION_BUILD_MISSILETURRET = 'buildmissileturret'
ACTION_BUILD_BUNKER = 'buildbunker'
ACTION_BUILD_FUSIONCORE = 'buildfusioncore'
ACTION_BUILD_GHOSTACADEMY = 'buildghostacademy'
ACTION_BUILD_BARRACKS = 'buildbarracks'                     # Selects SCV > builds barracks > sends SCV to harvest minerals
ACTION_BUILD_FACTORY = 'buildfactory'
ACTION_BUILD_STARPORT = 'buildstarport'
ACTION_BUILD_TECHLAB_BARRACKS = 'buildtechlabbarracks'
ACTION_BUILD_TECHLAB_FACTORY = 'buildtechlabfactory'
ACTION_BUILD_TECHLAB_STARPORT = 'buildtechlabstarport'
ACTION_BUILD_REACTOR_BARRACKS = 'buildreactorbarracks'
ACTION_BUILD_REACTOR_FACTORY = 'buildreactorfactory'
ACTION_BUILD_REACTOR_STARPORT = 'buildreactorstarport'

ACTION_RESEARCH_STIMPACK = 'researchstimpack'
ACTION_RESEARCH_COMBATSHIELD = 'researchcombatshield'
ACTION_RESEARCH_CONCUSSIVESHELL = 'researchconcussiveshell'

ACTION_EFFECT_STIMPACK = 'effectstimpack'

ACTION_TRAIN_SCV = 'trainscv'                               # Selects all command center > trains an scv > nothing
ACTION_TRAIN_MARINE = 'trainmarine'                         # Selects all barracks > trains marines > nothing
ACTION_TRAIN_MARAUDER = 'trainmarauder'

ACTION_ATTACK_ENEMY_BASE = 'attackenemybase'                                    # Selects army > attacks coordinates > nothing
ACTION_ATTACK_ENEMY_SECOND_BASE = 'attackenemysecondbase'
ACTION_ATTACK_MY_BASE = 'attackmybase'
ACTION_ATTACK_MY_SECOND_BASE = 'attackmysecondbase'


ACTION_HARVEST_MINERALS_IDLE = 'harvestmineralsidle'        # Selects random idle scv > sends him to harvest minerals
ACTION_HARVEST_MINERALS_FROM_GAS = 'harvestmineralsfromgas'
ACTION_HARVEST_GAS_FROM_MINERALS = 'harvestgasfromminerals'


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
        self.last_worker = sc2._NO_UNITS
        self.base_top_left = True

        '''
        We're defining names for our actions for two reasons:
        1) Abstraction: By defining names for our actions as strings we can pour in extra info. EX: The ACTION_ATTACK_x_y action contains
        can be parsed to retrieve (x, y) coordinates and pass them to the actual PySC2 action.
         
        2) Readability: Using names instead of literal numbers makes it easier to tell which action is which.
        '''
        self.named_actions = [
            ACTION_DO_NOTHING,

            # ACTION_BUILD_COMMAND_CENTER,
            ACTION_BUILD_SUPPLY_DEPOT,
            ACTION_BUILD_REFINERY,
            # ACTION_BUILD_ENGINEERINGBAY,
            # ACTION_BUILD_ARMORY,
            # ACTION_BUILD_MISSILETURRET,
            # ACTION_BUILD_BUNKER,
            # ACTION_BUILD_FUSIONCORE,
            # ACTION_BUILD_GHOSTACADEMY,
            ACTION_BUILD_BARRACKS,
            # ACTION_BUILD_FACTORY,
            # ACTION_BUILD_STARPORT,
            ACTION_BUILD_TECHLAB_BARRACKS,
            # ACTION_BUILD_TECHLAB_FACTORY,
            # ACTION_BUILD_TECHLAB_STARPORT,
            # ACTION_BUILD_REACTOR_BARRACKS,
            # ACTION_BUILD_REACTOR_FACTORY,
            # ACTION_BUILD_REACTOR_STARPORT,

            # ACTION_RESEARCH_STIMPACK,
            ACTION_RESEARCH_COMBATSHIELD,
            ACTION_RESEARCH_CONCUSSIVESHELL,

            ACTION_EFFECT_STIMPACK,

            ACTION_TRAIN_SCV,
            ACTION_TRAIN_MARINE,
            ACTION_TRAIN_MARAUDER,

            ACTION_HARVEST_MINERALS_IDLE,
            ACTION_HARVEST_MINERALS_FROM_GAS,
            ACTION_HARVEST_GAS_FROM_MINERALS,

            ACTION_ATTACK_ENEMY_BASE,
            ACTION_ATTACK_ENEMY_SECOND_BASE,
            ACTION_ATTACK_MY_BASE,
            ACTION_ATTACK_MY_SECOND_BASE,
        ]

        '''
        We're splitting the minimap into a 4x4 grid because the marine's effective range is able to cover
        the entire map from just this number of cells. For each (x, y) grid cell, we're defining an action called
        ACTION_ATTACK_x_y. When this actions is selected, we parse this string to retrieve this coordinate info
        and pass it as a parameter to the actual PySC2 action.
        '''
        # for mm_x in range(0, 64):
        #     for mm_y in range(0, 64):
        #         if (mm_x + 1) % 32 == 0 and (mm_y + 1) % 32 == 0:
        #             self.named_actions.append(ACTION_ATTACK + '_' + str(mm_x - 16) + '_' + str(mm_y - 16))
   
        '''
        In URNAI, the models can only return action indices. This index is passed by the agent to an ActionWrapper like this
        so that it decides what to do with the current action. For a complicated game like StarCraft 2 we can't just return an action,
        because most of them require extra parameters. So the get_action action of this wrapper is responsible for:
        1) Receiving an action index from the agent
        2) Selecting a named_action from the actions by using this index.
        3) Returning the PySC2 action that is equivalent to this named_action

        EX: 
        0) Agent receives action index 3 from its model
        1) Agent calls action_wrapper.get_action(3)
        2) get_action selects ACTION_BUILD_BARRACKS from its set of named actions
        3) get_action returns select_random_scv()
        '''
        self.action_indices = [idx for idx in range(len(self.named_actions))]


    def is_action_done(self):
        return self.move_number == 0

    
    def reset(self):
        self.move_number = 0


    def get_actions(self):
        return self.action_indices

    
    def split_action(self, smart_action):
        '''Breaks out (x, y) coordinates from a named action, if there are any.'''
        x = 0
        y = 0
        if '_' in smart_action:
            smart_action, x, y = smart_action.split('_')

        return (smart_action, x, y)


    def get_excluded_actions(self, obs):
        # supply_depot_count = get_units_amount(obs, units.Terran.SupplyDepot)

        # barracks_count = get_units_amount(obs, units.Terran.Barracks)

        # # Counts the amount of scvs currently on map
        # scv_count = get_units_amount(obs, units.Terran.SCV)

        # #supply_used = obs.player[3]
        # #supply_limit = obs.player[4]
        # supply_free = get_free_supply(obs)
        # army_supply = obs.player[5]
        # worker_supply = obs.player[6]

        # # Adding invalid actions to the list of excluded actions
        # excluded_actions = []
        # # If the supply depot limit of 2 was reached, removes the ability to build it.
        # if supply_depot_count == 4 or worker_supply == 0:
        #     excluded_actions.append(self.action_indices[1])
        # # If we have no supply depots or we have 2 barracks, we remove the ability to build barracks.
        # if supply_depot_count == 0 or barracks_count == 2 or worker_supply == 0:
        #     excluded_actions.append(self.action_indices[2])
        # # If we don't have any barracks or have reached supply limit, remove the ability to train marines
        # if supply_free == 0 or barracks_count == 0:
        #     excluded_actions.append(self.action_indices[4])
        # # If we have reached supply limit or amount of SCVs equal to 16, remove the ability to train SCVs
        # if supply_free == 0 or scv_count >= 16:
        #     excluded_actions.append(self.action_indices[5])
        # # If we have no marines, we remove attack actions
        # if army_supply == 0:
        #     excluded_actions.append(self.action_indices[6])
        #     excluded_actions.append(self.action_indices[7])
        #     excluded_actions.append(self.action_indices[8])
        #     excluded_actions.append(self.action_indices[9])
        
        excluded_actions = []

        return excluded_actions


    def get_action(self, action_idx, obs):
        named_action = self.named_actions[action_idx]
        named_action, x, y = self.split_action(named_action)

        if obs.game_loop[0] == 0:
            command_center = get_my_units_by_type(obs, units.Terran.CommandCenter)[0]
            self.base_top_left = (command_center.x < 32)
        
        '''LIST OF ACTIONS THE AGENT IS ABLE TO CHOOSE FROM:'''

        # BUILD COMMAND CENTER
        if named_action == ACTION_BUILD_COMMAND_CENTER:
            if get_units_amount(obs, units.Terran.CommandCenter) < 2:
                if self.move_number == 0:
                    self.move_number += 1
                    x = random.randint(0,63)
                    y = random.randint(0,63)
                    target = [x, y]
                    action, self.last_worker = build_structure_by_type(obs, sc2._BUILD_COMMAND_CENTER, target)
                    return action
                if self.move_number == 1:
                    self.move_number +=1
                    return harvest_gather_minerals_quick(obs, self.last_worker)
                if self.move_number == 2:
                    self.move_number = 0

        # BUILD SUPPLY DEPOT
        if named_action == ACTION_BUILD_SUPPLY_DEPOT:
            if get_units_amount(obs, units.Terran.SupplyDepot) < 10:
                if self.move_number == 0:
                    self.move_number += 1
                    x = random.randint(0,63)
                    y = random.randint(0,63)
                    target = [x, y]
                    action, self.last_worker = build_structure_by_type(obs, sc2._BUILD_SUPPLY_DEPOT, target)
                    return action
                if self.move_number == 1:
                    self.move_number +=1
                    return harvest_gather_minerals_quick(obs, self.last_worker)
                if self.move_number == 2:
                    self.move_number = 0

        # BUILD REFINERY
        if named_action == ACTION_BUILD_REFINERY:
            if get_units_amount(obs, units.Terran.Refinery) < 8:
                if self.move_number == 0:
                    self.move_number += 1
                    # Using our get_exploitable_geyser function defined int actions\sc2.py to choose an available Vespene Geyser to build our refinery
                    chosen_geyser = get_exploitable_geyser(obs, sc2_env.Race.terran)
                    # Building a refinery in the chosen geyser
                    action, self.last_worker = build_structure_by_type(obs, sc2._BUILD_REFINERY, chosen_geyser)
                    return action
                if self.move_number == 1:
                    self.move_number +=1
                    return harvest_gather_minerals_quick(obs, self.last_worker)
                if self.move_number == 2:
                    self.move_number = 0

        # BUILD ENGINEERINGBAY
        if named_action == ACTION_BUILD_ENGINEERINGBAY:
            if get_units_amount(obs, units.Terran.EngineeringBay) < 1:
                if self.move_number == 0:
                    self.move_number += 1
                    x = random.randint(0,63)
                    y = random.randint(0,63)
                    target = [x, y]
                    action, self.last_worker = build_structure_by_type(obs, sc2._BUILD_ENGINEERINGBAY, target)
                    return action
                if self.move_number == 1:
                    self.move_number +=1
                    return harvest_gather_minerals_quick(obs, self.last_worker)
                if self.move_number == 2:
                    self.move_number = 0

        # BUILD ARMORY
        if named_action == ACTION_BUILD_ARMORY:
            if get_units_amount(obs, units.Terran.Armory) < 1:
                if self.move_number == 0:
                    self.move_number += 1
                    x = random.randint(0,63)
                    y = random.randint(0,63)
                    target = [x, y]
                    action, self.last_worker = build_structure_by_type(obs, sc2._BUILD_ARMORY, target)
                    return action
                if self.move_number == 1:
                    self.move_number +=1
                    return harvest_gather_minerals_quick(obs, self.last_worker)
                if self.move_number == 2:
                    self.move_number = 0

        # BUILD MISSILE TURRET
        if named_action == ACTION_BUILD_MISSILETURRET:
            if get_units_amount(obs, units.Terran.MissileTurret) < 1:
                if self.move_number == 0:
                    self.move_number += 1
                    x = random.randint(0,63)
                    y = random.randint(0,63)
                    target = [x, y]
                    action, self.last_worker = build_structure_by_type(obs, sc2._BUILD_MISSILETURRET, target)
                    return action
                if self.move_number == 1:
                    self.move_number +=1
                    return harvest_gather_minerals_quick(obs, self.last_worker)
                if self.move_number == 2:
                    self.move_number = 0

        # BUILD BUNKER
        if named_action == ACTION_BUILD_BUNKER:
            if get_units_amount(obs, units.Terran.Bunker) < 1:
                if self.move_number == 0:
                    self.move_number += 1
                    x = random.randint(0,63)
                    y = random.randint(0,63)
                    target = [x, y]
                    action, self.last_worker = build_structure_by_type(obs, sc2._BUILD_BUNKER, target)
                    return action
                if self.move_number == 1:
                    self.move_number +=1
                    return harvest_gather_minerals_quick(obs, self.last_worker)
                if self.move_number == 2:
                    self.move_number = 0

        # BUILD FUSIONCORE
        if named_action == ACTION_BUILD_FUSIONCORE:
            if get_units_amount(obs, units.Terran.FusionCore) < 1:
                if self.move_number == 0:
                    self.move_number += 1
                    x = random.randint(0,63)
                    y = random.randint(0,63)
                    target = [x, y]
                    action, self.last_worker = build_structure_by_type(obs, sc2._BUILD_FUSIONCORE, target)
                    return action
                if self.move_number == 1:
                    self.move_number +=1
                    return harvest_gather_minerals_quick(obs, self.last_worker)
                if self.move_number == 2:
                    self.move_number = 0

        # BUILD GHOSTACADEMY
        if named_action == ACTION_BUILD_GHOSTACADEMY:
            if get_units_amount(obs, units.Terran.GhostAcademy) < 1:
                if self.move_number == 0:
                    self.move_number += 1
                    x = random.randint(0,63)
                    y = random.randint(0,63)
                    target = [x, y]
                    action, self.last_worker = build_structure_by_type(obs, sc2._BUILD_GHOSTACADEMY, target)
                    return action
                if self.move_number == 1:
                    self.move_number +=1
                    return harvest_gather_minerals_quick(obs, self.last_worker)
                if self.move_number == 2:
                    self.move_number = 0

        # BUILD BARRACKS
        if named_action == ACTION_BUILD_BARRACKS:
            if get_units_amount(obs, units.Terran.Barracks) < 3:
                if self.move_number == 0:
                    self.move_number += 1
                    x = random.randint(0,63)
                    y = random.randint(0,63)
                    target = [x, y]
                    action, self.last_worker = build_structure_by_type(obs, sc2._BUILD_BARRACKS, target)
                    return action
                if self.move_number == 1:
                    self.move_number +=1
                    return harvest_gather_minerals_quick(obs, self.last_worker)
                if self.move_number == 2:
                    self.move_number = 0

        # BUILD FACTORY
        if named_action == ACTION_BUILD_FACTORY:
            if get_units_amount(obs, units.Terran.Factory) < 2:
                if self.move_number == 0:
                    self.move_number += 1
                    x = random.randint(0,63)
                    y = random.randint(0,63)
                    target = [x, y]
                    action, self.last_worker = build_structure_by_type(obs, sc2._BUILD_FACTORY, target)
                    return action
                if self.move_number == 1:
                    self.move_number +=1
                    return harvest_gather_minerals_quick(obs, self.last_worker)
                if self.move_number == 2:
                    self.move_number = 0

        # BUILD STARPORT
        if named_action == ACTION_BUILD_STARPORT:
            if get_units_amount(obs, units.Terran.Starport) < 2:
                if self.move_number == 0:
                    self.move_number += 1
                    x = random.randint(0,63)
                    y = random.randint(0,63)
                    target = [x, y]
                    action, self.last_worker = build_structure_by_type(obs, sc2._BUILD_STARPORT, target)
                    return action
                if self.move_number == 1:
                    self.move_number +=1
                    return harvest_gather_minerals_quick(obs, self.last_worker)
                if self.move_number == 2:
                    self.move_number = 0

        # BUILD TECHLAB BARRACKS
        if named_action == ACTION_BUILD_TECHLAB_BARRACKS:
            if self.move_number == 0:
                self.move_number += 1

                barracks = get_my_units_by_type(obs, units.Terran.Barracks)
                if len(barracks) > 0:
                    target = random.choice(barracks)
                    action, self.last_worker = build_structure_by_type(obs, sc2._BUILD_TECHLAB_BARRACKS, target)
                    return action
                return no_op()
            if self.move_number == 1:
                self.move_number +=1
                return harvest_gather_minerals_quick(obs, self.last_worker)
            if self.move_number == 2:
                self.move_number = 0

        # BUILD TECHLAB FACTORY
        if named_action == ACTION_BUILD_TECHLAB_FACTORY:
            if self.move_number == 0:
                self.move_number += 1

                factories = get_my_units_by_type(obs, units.Terran.Factory)
                if len(factories) > 0:
                    target = random.choice(factories)
                    action, self.last_worker = build_structure_by_type(obs, sc2._BUILD_TECHLAB_FACTORY, target)
                    return action
                return no_op()
            if self.move_number == 1:
                self.move_number +=1
                return harvest_gather_minerals_quick(obs, self.last_worker)
            if self.move_number == 2:
                self.move_number = 0

        # BUILD TECHLAB STARPORT
        if named_action == ACTION_BUILD_TECHLAB_STARPORT:
            if self.move_number == 0:
                self.move_number += 1

                starports = get_my_units_by_type(obs, units.Terran.Starport)
                if len(starports) > 0:
                    target = random.choice(starports)
                    action, self.last_worker = build_structure_by_type(obs, sc2._BUILD_TECHLAB_STARPORT, target)
                    return action
                return no_op()
            if self.move_number == 1:
                self.move_number +=1
                return harvest_gather_minerals_quick(obs, self.last_worker)
            if self.move_number == 2:
                self.move_number = 0

        # BUILD REACTOR BARRACKS
        if named_action == ACTION_BUILD_REACTOR_BARRACKS:
            if self.move_number == 0:
                self.move_number += 1

                barracks = get_my_units_by_type(obs, units.Terran.Barracks)
                if len(barracks) > 0:
                    target = random.choice(barracks)
                    action, self.last_worker = build_structure_by_type(obs, sc2._BUILD_REACTOR_BARRACKS, target)
                    return action
                return no_op()
            if self.move_number == 1:
                self.move_number +=1
                return harvest_gather_minerals_quick(obs, self.last_worker)
            if self.move_number == 2:
                self.move_number = 0

        # BUILD REACTOR FACTORY
        if named_action == ACTION_BUILD_REACTOR_FACTORY:
            if self.move_number == 0:
                self.move_number += 1

                factories = get_my_units_by_type(obs, units.Terran.Factory)
                if len(factories) > 0:
                    target = random.choice(factories)
                    action, self.last_worker = build_structure_by_type(obs, sc2._BUILD_REACTOR_FACTORY, target)
                    return action
                return no_op()
            if self.move_number == 1:
                self.move_number +=1
                return harvest_gather_minerals_quick(obs, self.last_worker)
            if self.move_number == 2:
                self.move_number = 0

        # BUILD REACTOR STARPORT
        if named_action == ACTION_BUILD_REACTOR_STARPORT:
            if self.move_number == 0:
                self.move_number += 1

                starports = get_my_units_by_type(obs, units.Terran.Starport)
                if len(starports) > 0:
                    target = random.choice(starports)
                    action, self.last_worker = build_structure_by_type(obs, sc2._BUILD_REACTOR_STARPORT, target)
                    return action
                return no_op()
            if self.move_number == 1:
                self.move_number +=1
                return harvest_gather_minerals_quick(obs, self.last_worker)
            if self.move_number == 2:
                self.move_number = 0
                

        # HARVEST MINERALS WITH IDLE WORKER
        if named_action == ACTION_HARVEST_MINERALS_IDLE:
            idle_worker = select_idle_worker(obs, sc2_env.Race.terran)
            if idle_worker != sc2._NO_UNITS:
                if building_exists(obs, units.Terran.CommandCenter):
                    ccs = get_my_units_by_type(obs, units.Terran.CommandCenter)
                    for cc in ccs:
                        if get_euclidean_distance([idle_worker.x, idle_worker.y], [cc.x, cc.y]) < 10:
                            return harvest_gather_minerals(obs, idle_worker, cc)
            return no_op()

        # TO DO: Create a harvest minerals with worker from refinery line so the bot can juggle workers from mineral lines to gas back and forth

        # HARVEST MINERALS WITH WORKER FROM GAS LINE
        if named_action == ACTION_HARVEST_MINERALS_FROM_GAS:
            if building_exists(obs, units.Terran.CommandCenter):
                ccs = get_my_units_by_type(obs, units.Terran.CommandCenter)
                for cc in ccs:
                    # Check if command center is not full of workers yet
                    if cc.assigned_harvesters < cc.ideal_harvesters:
                        workers = get_my_units_by_type(obs, units.Terran.SCV)
                        for worker in workers:
                            if get_euclidean_distance([worker.x, worker.y], [cc.x, cc.y]) < 10:
                                # Checking if worker is harvesting, if so, send him to harvest gas
                                if worker.order_id_0 == 362 or worker.order_id_0 == 359:
                                    return harvest_gather_minerals(obs, worker, cc)
            return no_op()

        # HARVEST GAS WITH WORKER FROM MINERAL LINE
        if named_action == ACTION_HARVEST_GAS_FROM_MINERALS:
            if building_exists(obs, units.Terran.CommandCenter):
                if building_exists(obs, units.Terran.Refinery):
                    refineries = get_my_units_by_type(obs, units.Terran.Refinery)
                    # Going through all refineries
                    for refinery in refineries:
                        # Checking if refinery is not full of workers yet
                        if refinery.assigned_harvesters < refinery.ideal_harvesters:
                            workers = get_my_units_by_type(obs, units.Terran.SCV)
                            for worker in workers:
                                # Checking if worker is close by to the refinery
                                if get_euclidean_distance([worker.x, worker.y], [refinery.x, refinery.y]) < 10:
                                    # Checking if worker is harvesting, if so, send him to harvest gas
                                    if worker.order_id_0 == 362 or worker.order_id_0 == 359:
                                        return harvest_gather_gas(obs, worker, refinery)
            return no_op()

        # RESEARCH STIMPACK
        if named_action == ACTION_RESEARCH_STIMPACK:
            return research_upgrade(obs, sc2._RESEARCH_TERRAN_STIMPACK, units.Terran.BarracksTechLab)

        # RESEARCH COMBATSHIELD
        if named_action == ACTION_RESEARCH_COMBATSHIELD:
            return research_upgrade(obs, sc2._RESEARCH_TERRAN_COMBATSHIELD, units.Terran.BarracksTechLab)

        # RESEARCH CONCUSSIVESHELL
        if named_action == ACTION_RESEARCH_CONCUSSIVESHELL:
            return research_upgrade(obs, sc2._RESEARCH_TERRAN_CONCUSSIVESHELL, units.Terran.BarracksTechLab)

        # EFFECT STIMPACK
        if named_action == ACTION_EFFECT_STIMPACK:
            army = []
            marines = get_my_units_by_type(obs, units.Terran.Marine)
            marauders = get_my_units_by_type(obs, units.Terran.Marauder)
            army.extend(marines)
            army.extend(marauders)
            if building_exists(obs, units.Terran.BarracksTechLab):
                if len(army) > 0:
                    return effect_units(obs, sc2._EFFECT_STIMPACK, army)
            return no_op()

        # TRAIN SCV
        if named_action == ACTION_TRAIN_SCV:
            return train_unit(obs, sc2._TRAIN_SCV, units.Terran.CommandCenter)

        # TRAIN MARINE
        if named_action == ACTION_TRAIN_MARINE:
            return train_unit(obs, sc2._TRAIN_MARINE, units.Terran.Barracks)

        # TRAIN MARAUDER
        if named_action == ACTION_TRAIN_MARAUDER:
            return train_unit(obs, sc2._TRAIN_MARAUDER, units.Terran.Barracks)

        # ATTACK ENEMY BASE
        if named_action == ACTION_ATTACK_ENEMY_BASE:
            attack_xy = (38, 44) if self.base_top_left else (19, 23)
            x_offset = random.randint(-4, 4)
            y_offset = random.randint(-4, 4)
            army = []
            marines = get_my_units_by_type(obs, units.Terran.Marine)
            marauders = get_my_units_by_type(obs, units.Terran.Marauder)
            army.extend(marines)
            army.extend(marauders)
            if len(army) > 0:
                target = [attack_xy[0] + x_offset, attack_xy[1] + y_offset]
                return attack_target_point(obs, army, target)
            return no_op()

        # ATTACK ENEMY SECOND BASE
        if named_action == ACTION_ATTACK_ENEMY_SECOND_BASE:
            attack_xy = (19, 44) if self.base_top_left else (38, 23)
            x_offset = random.randint(-4, 4)
            y_offset = random.randint(-4, 4)
            army = []
            marines = get_my_units_by_type(obs, units.Terran.Marine)
            marauders = get_my_units_by_type(obs, units.Terran.Marauder)
            army.extend(marines)
            army.extend(marauders)
            if len(army) > 0:
                target = [attack_xy[0] + x_offset, attack_xy[1] + y_offset]
                return attack_target_point(obs, army, target)
            return no_op()

        # ATTACK MY BASE
        if named_action == ACTION_ATTACK_MY_BASE:
            attack_xy = (19, 23) if self.base_top_left else (38, 44)
            x_offset = random.randint(-4, 4)
            y_offset = random.randint(-4, 4)
            army = []
            marines = get_my_units_by_type(obs, units.Terran.Marine)
            marauders = get_my_units_by_type(obs, units.Terran.Marauder)
            army.extend(marines)
            army.extend(marauders)
            if len(army) > 0:
                target = [attack_xy[0] + x_offset, attack_xy[1] + y_offset]
                return attack_target_point(obs, army, target)
            return no_op()

        # ATTACK MY SECOND BASE
        if named_action == ACTION_ATTACK_MY_SECOND_BASE:
            attack_xy = (38, 23) if self.base_top_left else (19, 44)
            x_offset = random.randint(-4, 4)
            y_offset = random.randint(-4, 4)
            army = []
            marines = get_my_units_by_type(obs, units.Terran.Marine)
            marauders = get_my_units_by_type(obs, units.Terran.Marauder)
            army.extend(marines)
            army.extend(marauders)
            if len(army) > 0:
                target = [attack_xy[0] + x_offset, attack_xy[1] + y_offset]
                return attack_target_point(obs, army, target)
            return no_op()


        return no_op()