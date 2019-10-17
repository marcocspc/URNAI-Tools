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
ACTION_BUILD_SENSORTOWER = 'buildsensortower'
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

ACTION_RESEARCH_INF_WEAPONS = 'researchinfantryweapons'
ACTION_RESEARCH_INF_ARMOR = 'researchinfantryarmor'
ACTION_RESEARCH_HISEC_AUTOTRACKING = 'researchhisecautotracking'
ACTION_RESEARCH_NEOSTEEL_FRAME = 'researchneosteelframe'
ACTION_RESEARCH_STRUCTURE_ARMOR = 'researchstructurearmor'

ACTION_RESEARCH_SHIPS_WEAPONS = 'researchshipsweapons'
ACTION_RESEARCH_VEHIC_WEAPONS = 'researchvehicweapons'
ACTION_RESEARCH_SHIPVEHIC_PLATES = 'researchshipvehicplates'

ACTION_RESEARCH_GHOST_CLOAK = 'researchghostcloak'

ACTION_RESEARCH_STIMPACK = 'researchstimpack'
ACTION_RESEARCH_COMBATSHIELD = 'researchcombatshield'
ACTION_RESEARCH_CONCUSSIVESHELL = 'researchconcussiveshell'

ACTION_RESEARCH_INFERNAL_PREIGNITER = 'researchinfernalpreigniter'
ACTION_RESEARCH_DRILLING_CLAWS = 'researchdrillingclaws'
ACTION_RESEARCH_CYCLONE_LOCKONDMG = 'researchcyclonelockondmg'
ACTION_RESEARCH_CYCLONE_RAPIDFIRE = 'researchcyclonerapidfire'

ACTION_RESEARCH_HIGHCAPACITYFUEL = 'researchhighcapacityfuel'
ACTION_RESEARCH_CORVIDREACTOR = 'researchcorvidreactor'
ACTION_RESEARCH_BANSHEECLOAK = 'researchbansheecloak'
ACTION_RESEARCH_BANSHEEHYPERFLIGHT = 'researchbansheehyperflight'
ACTION_RESEARCH_ADVANCEDBALLISTICS = 'researchadvancedballistics'

ACTION_RESEARCH_BATTLECRUISER_WEAPONREFIT = 'researchbattlecruiserweaponrefit'

ACTION_EFFECT_STIMPACK = 'effectstimpack'

ACTION_TRAIN_SCV = 'trainscv'                               # Selects all command center > trains an scv > nothing
ACTION_TRAIN_MARINE = 'trainmarine'                         # Selects all barracks > trains marines > nothing
ACTION_TRAIN_MARAUDER = 'trainmarauder'
ACTION_TRAIN_REAPER = 'trainreaper'
ACTION_TRAIN_GHOST = 'trainghost'
ACTION_TRAIN_HELLION = 'trainhellion'
ACTION_TRAIN_HELLBAT = 'trainhellbat'
ACTION_TRAIN_SIEGETANK = 'trainsiegetank'
ACTION_TRAIN_CYCLONE = 'traincyclone'
ACTION_TRAIN_WIDOWMINE = 'trainwidowmine'
ACTION_TRAIN_THOR = 'trainthor'
ACTION_TRAIN_VIKING = 'trainviking'
ACTION_TRAIN_MEDIVAC = 'trainmedivac'
ACTION_TRAIN_LIBERATOR = 'trainliberator'
ACTION_TRAIN_RAVEN = 'trainraven'
ACTION_TRAIN_BANSHEE = 'trainbanshee'
ACTION_TRAIN_BATTLECRUISER = 'trainbattlecruiser'


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
        self.units_to_attack = sc2._NO_UNITS
        self.last_attack_action = 0
        self.base_top_left = True

        '''
        We're defining names for our actions for two reasons:
        1) Abstraction: By defining names for our actions as strings we can pour in extra info. EX: The ACTION_ATTACK_x_y action contains
        can be parsed to retrieve (x, y) coordinates and pass them to the actual PySC2 action.
         
        2) Readability: Using names instead of literal numbers makes it easier to tell which action is which.
        '''
        self.named_actions = [
            ACTION_DO_NOTHING,

            ACTION_BUILD_COMMAND_CENTER,
            ACTION_BUILD_SUPPLY_DEPOT,
            ACTION_BUILD_REFINERY,
            ACTION_BUILD_ENGINEERINGBAY,
            ACTION_BUILD_ARMORY,
            ACTION_BUILD_MISSILETURRET,
            ACTION_BUILD_SENSORTOWER,
            ACTION_BUILD_BUNKER,
            ACTION_BUILD_FUSIONCORE,
            ACTION_BUILD_GHOSTACADEMY,
            ACTION_BUILD_BARRACKS,
            ACTION_BUILD_FACTORY,
            ACTION_BUILD_STARPORT,
            ACTION_BUILD_TECHLAB_BARRACKS,
            ACTION_BUILD_TECHLAB_FACTORY,
            ACTION_BUILD_TECHLAB_STARPORT,
            ACTION_BUILD_REACTOR_BARRACKS,
            ACTION_BUILD_REACTOR_FACTORY,
            ACTION_BUILD_REACTOR_STARPORT,

            # ENGINEERING BAY RESEARCH
            # ACTION_RESEARCH_INF_WEAPONS,
            # ACTION_RESEARCH_INF_ARMOR,
            # ACTION_RESEARCH_HISEC_AUTOTRACKING,
            # ACTION_RESEARCH_NEOSTEEL_FRAME,
            # ACTION_RESEARCH_STRUCTURE_ARMOR,
            
            # ARMORY RESEARCH
            # ACTION_RESEARCH_SHIPS_WEAPONS,
            # ACTION_RESEARCH_VEHIC_WEAPONS,
            # ACTION_RESEARCH_SHIPVEHIC_PLATES,

            # GHOST ACADEMY RESEARCH
            # ACTION_RESEARCH_GHOST_CLOAK,

            # BARRACKS RESEARCH
            ACTION_RESEARCH_STIMPACK,
            ACTION_RESEARCH_COMBATSHIELD,
            ACTION_RESEARCH_CONCUSSIVESHELL,

            # FACTORY RESEARCH
            # ACTION_RESEARCH_INFERNAL_PREIGNITER,
            # ACTION_RESEARCH_DRILLING_CLAWS,
            # ACTION_RESEARCH_CYCLONE_LOCKONDMG,
            # ACTION_RESEARCH_CYCLONE_RAPIDFIRE,

            # STARPORT RESEARCH
            # ACTION_RESEARCH_HIGHCAPACITYFUEL,
            # ACTION_RESEARCH_CORVIDREACTOR,
            # ACTION_RESEARCH_BANSHEECLOAK,
            # ACTION_RESEARCH_BANSHEEHYPERFLIGHT,
            # ACTION_RESEARCH_ADVANCEDBALLISTICS,

            # FUSION CORE RESEARCH
            # ACTION_RESEARCH_BATTLECRUISER_WEAPONREFIT,

            # ACTION_EFFECT_STIMPACK,

            ACTION_TRAIN_SCV,

            ACTION_TRAIN_MARINE,
            ACTION_TRAIN_MARAUDER,
            # ACTION_TRAIN_REAPER,
            # ACTION_TRAIN_GHOST,

            # ACTION_TRAIN_HELLION,
            # ACTION_TRAIN_HELLBAT,
            ACTION_TRAIN_SIEGETANK,
            ACTION_TRAIN_CYCLONE,
            # ACTION_TRAIN_WIDOWMINE,
            # ACTION_TRAIN_THOR,

            ACTION_TRAIN_VIKING,
            ACTION_TRAIN_MEDIVAC,
            # ACTION_TRAIN_LIBERATOR,
            # ACTION_TRAIN_RAVEN,
            # ACTION_TRAIN_BANSHEE,
            # ACTION_TRAIN_BATTLECRUISER,


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

        excluded_actions = []

        excluded_actions = self.named_actions.copy()

        # ACTION_BUILD_COMMAND_CENTER CHECK
        if building_exists(obs, units.Terran.SCV) and obs.player.minerals > 400:
            excluded_actions.remove(ACTION_BUILD_COMMAND_CENTER)
        # ACTION_BUILD_SUPPLY_DEPOT CHECK
        if building_exists(obs, units.Terran.SCV) and obs.player.minerals > 100:
            excluded_actions.remove(ACTION_BUILD_SUPPLY_DEPOT)
        # ACTION_BUILD_REFINERY CHECK
        if building_exists(obs, units.Terran.SCV) and obs.player.minerals > 75:
            excluded_actions.remove(ACTION_BUILD_REFINERY)

        # ACTIONS DEPENDENT ON A SUPPLY DEPOT
        if building_exists(obs, units.Terran.SupplyDepot):
            # ACTION_BUILD_BARRACKS CHECK
            if building_exists(obs, units.Terran.SCV) and obs.player.minerals > 150:
                excluded_actions.remove(ACTION_BUILD_BARRACKS)

        # ACTIONS DEPENDENT ON A BARRACKS
        if building_exists(obs, units.Terran.Barracks):
            # ACTION_BUILD_BUNKER CHECK
            if building_exists(obs, units.Terran.SCV) and obs.player.minerals > 100:
                excluded_actions.remove(ACTION_BUILD_BUNKER)
            '''# ACTION_BUILD_ORBITAL_COMMAND CHECK
            if building_exists(obs, units.Terran.SCV) and obs.player.minerals > 550:
                excluded_actions.remove(ACTION_BUILD_ORBITAL_COMMAND)'''
            # ACTION_BUILD_FACTORY CHECK
            if building_exists(obs, units.Terran.SCV) and obs.player.minerals > 150 and obs.player.vespene > 100:
                excluded_actions.remove(ACTION_BUILD_FACTORY)
            # ACTION_BUILD_GHOSTACADEMY CHECK
            if building_exists(obs, units.Terran.SCV) and obs.player.minerals > 150 and obs.player.vespene > 50:
                excluded_actions.remove(ACTION_BUILD_GHOSTACADEMY)
            # ACTION_BUILD_TECHLAB_BARRACKS CHECK
            if building_exists(obs, units.Terran.SCV) and obs.player.minerals > 50 and obs.player.vespene > 25:
                excluded_actions.remove(ACTION_BUILD_TECHLAB_BARRACKS)
            # ACTION_BUILD_REACTOR_BARRACKS CHECK
            if building_exists(obs, units.Terran.SCV) and obs.player.minerals > 50 and obs.player.vespene > 50:
                excluded_actions.remove(ACTION_BUILD_REACTOR_BARRACKS)

        # ACTIONS DEPENDENT ON A FACTORY
        if building_exists(obs, units.Terran.Factory):
            # ACTION_BUILD_ARMORY CHECK
            if building_exists(obs, units.Terran.SCV) and obs.player.minerals > 150 and obs.player.vespene > 100:
                excluded_actions.remove(ACTION_BUILD_ARMORY)
            # ACTION_BUILD_STARPORT CHECK
            if building_exists(obs, units.Terran.SCV) and obs.player.minerals > 150 and obs.player.vespene > 100:
                excluded_actions.remove(ACTION_BUILD_STARPORT)
            # ACTION_BUILD_TECHLAB_FACTORY CHECK
            if building_exists(obs, units.Terran.SCV) and obs.player.minerals > 50 and obs.player.vespene > 25:
                excluded_actions.remove(ACTION_BUILD_TECHLAB_FACTORY)
            # ACTION_BUILD_REACTOR_FACTORY CHECK
            if building_exists(obs, units.Terran.SCV) and obs.player.minerals > 50 and obs.player.vespene > 50:
                excluded_actions.remove(ACTION_BUILD_REACTOR_FACTORY)

        if building_exists(obs, units.Terran.Starport):
            # ACTION_BUILD_FUSIONCORE CHECK
            if building_exists(obs, units.Terran.SCV) and obs.player.minerals > 150 and obs.player.vespene > 150:
                excluded_actions.remove(ACTION_BUILD_FUSIONCORE)
            # ACTION_BUILD_TECHLAB_STARPORT CHECK
            if building_exists(obs, units.Terran.SCV) and obs.player.minerals > 50 and obs.player.vespene > 25:
                excluded_actions.remove(ACTION_BUILD_TECHLAB_STARPORT)
            # ACTION_BUILD_REACTOR_STARPORT CHECK
            if building_exists(obs, units.Terran.SCV) and obs.player.minerals > 50 and obs.player.vespene > 50:
                excluded_actions.remove(ACTION_BUILD_REACTOR_STARPORT)
                
        if building_exists(obs, units.Terran.CommandCenter) or \
            building_exists(obs, units.Terran.PlanetaryFortress) or \
            building_exists(obs, units.Terran.OrbitalCommand):

            # ACTION_BUILD_ENGINEERINGBAY CHECK
            if building_exists(obs, units.Terran.SCV) and obs.player.minerals > 125:
                excluded_actions.remove(ACTION_BUILD_ENGINEERINGBAY)

        if building_exists(obs, units.Terran.EngineeringBay):
            '''# ACTION_BUILD_ORBITAL_COMMAND CHECK
            if building_exists(obs, units.Terran.SCV) and obs.player.minerals > 550 and obs.player.vespene > 150:
                excluded_actions.remove(ACTION_BUILD_ORBITAL_COMMAND)'''
            # ACTION_BUILD_SENSORTOWER CHECK
            if building_exists(obs, units.Terran.SCV) and obs.player.minerals > 125:
                excluded_actions.remove(ACTION_BUILD_SENSORTOWER)
            # ACTION_BUILD_MISSILETURRET CHECK
            if building_exists(obs, units.Terran.SCV) and obs.player.minerals > 100:
                excluded_actions.remove(ACTION_BUILD_MISSILETURRET)


        # if not building_exists(obs, units.Terran.SupplyDepot):
        #     # Building actions dependent on a supply depot
        #     excluded_actions.append(self.action_indices[self.named_actions.index(ACTION_BUILD_BARRACKS)])
        #     excluded_actions.append(self.action_indices[self.named_actions.index(ACTION_BUILD_FACTORY)])
        #     #excluded_actions.append(self.action_indices[self.named_actions.index(ACTION_BUILD_BUNKER)])
        #     #excluded_actions.append(self.action_indices[self.named_actions.index(ACTION_BUILD_GHOSTACADEMY)])
        #     excluded_actions.append(self.action_indices[self.named_actions.index(ACTION_BUILD_ARMORY)])
        #     # excluded_actions.append(self.action_indices[self.named_actions.index(ACTION_BUILD_STARPORT)])
        #     # excluded_actions.append(self.action_indices[self.named_actions.index(ACTION_BUILD_FUSIONCORE)])

        # if not building_exists(obs, units.Terran.Barracks):
        #     # Building actions dependent on a barracks
        #     excluded_actions.append(self.action_indices[self.named_actions.index(ACTION_BUILD_FACTORY)])
        #     # excluded_actions.append(self.action_indices[self.named_actions.index(ACTION_BUILD_BUNKER)])
        #     # excluded_actions.append(self.action_indices[self.named_actions.index(ACTION_BUILD_GHOSTACADEMY)])
        #     excluded_actions.append(self.action_indices[self.named_actions.index(ACTION_BUILD_ARMORY)])
        #     # excluded_actions.append(self.action_indices[self.named_actions.index(ACTION_BUILD_STARPORT)])
        #     # excluded_actions.append(self.action_indices[self.named_actions.index(ACTION_BUILD_FUSIONCORE)])
        #     # excluded_actions.append(self.action_indices[self.named_actions.index(ACTION_BUILD_TECHLAB_BARRACKS)]
        #     # excluded_actions.append(self.action_indices[self.named_actions.index(ACTION_BUILD_REACTOR_BARRACKS)])
        #     # Training actions dependent on a barracks
        #     excluded_actions.append(self.action_indices[self.named_actions.index(ACTION_TRAIN_MARINE)])
        #     excluded_actions.append(self.action_indices[self.named_actions.index(ACTION_TRAIN_MARAUDER)])
        #     # excluded_actions.append(self.action_indices[self.named_actions.index(ACTION_TRAIN_REAPER)])
        #     # excluded_actions.append(self.action_indices[self.named_actions.index(ACTION_TRAIN_GHOST)])
        #     # Research actions dependent on a barracks
        #     excluded_actions.append(self.action_indices[self.named_actions.index(ACTION_RESEARCH_STIMPACK)])
        #     excluded_actions.append(self.action_indices[self.named_actions.index(ACTION_RESEARCH_COMBATSHIELD)])
        #     excluded_actions.append(self.action_indices[self.named_actions.index(ACTION_RESEARCH_CONCUSSIVESHELL)])


        # if not building_exists(obs, units.Terran.Factory):
        #     excluded_actions.append(self.action_indices[self.named_actions.index(ACTION_BUILD_ARMORY)])
        #     # excluded_actions.append(self.action_indices[self.named_actions.index(ACTION_BUILD_STARPORT)])
        #     # excluded_actions.append(self.action_indices[self.named_actions.index(ACTION_BUILD_FUSIONCORE)])
        
        # if not building_exists(obs, units.Terran.CommandCenter):
        #     excluded_actions.append(self.action_indices[self.named_actions.index(ACTION_BUILD_ENGINEERINGBAY)])
        #     # excluded_actions.append(self.action_indices[self.named_actions.index(ACTION_BUILD_MISSILETURRET)])

        # # if not building_exists(obs, units.Terran.EngineeringBay):
        #     # excluded_actions.append(self.action_indices[self.named_actions.index(ACTION_BUILD_MISSILETURRET)])

        # if obs.player.food_army == 0:
        #     excluded_actions.append(self.action_indices[self.named_actions.index(ACTION_ATTACK_ENEMY_BASE)])
        #     excluded_actions.append(self.action_indices[self.named_actions.index(ACTION_ATTACK_ENEMY_SECOND_BASE)])
        #     excluded_actions.append(self.action_indices[self.named_actions.index(ACTION_ATTACK_MY_BASE)])
        #     excluded_actions.append(self.action_indices[self.named_actions.index(ACTION_ATTACK_MY_SECOND_BASE)])

        # if get_units_amount(obs, units.Terran.SCV) == 0:
        #     excluded_actions.append(self.action_indices[self.named_actions.index(ACTION_HARVEST_MINERALS_FROM_GAS)])
        #     excluded_actions.append(self.action_indices[self.named_actions.index(ACTION_HARVEST_GAS_FROM_MINERALS)])
        #     excluded_actions.append(self.action_indices[self.named_actions.index(ACTION_HARVEST_MINERALS_IDLE)])
        
        return excluded_actions


    def get_action(self, action_idx, obs):
        named_action = self.named_actions[action_idx]
        named_action, x, y = self.split_action(named_action)

        if self.units_to_attack != sc2._NO_UNITS:
            named_action = self.last_attack_action

        if obs.game_loop[0] == 0:
            command_center = get_my_units_by_type(obs, units.Terran.CommandCenter)[0]
            self.base_top_left = (command_center.x < 32)

        if self.base_top_left:
            ybrange = 0
            ytrange = 32
        else:
            ybrange = 32
            ytrange = 63

        '''LIST OF ACTIONS THE AGENT IS ABLE TO CHOOSE FROM:'''

        # BUILD COMMAND CENTER
        if named_action == ACTION_BUILD_COMMAND_CENTER:
            action, self.last_worker, self.move_number = build_structure_raw_pt(obs, units.Terran.CommandCenter, sc2._BUILD_COMMAND_CENTER, self.move_number, self.last_worker, self.base_top_left, max_amount = 2)
            return action

        # BUILD SUPPLY DEPOT
        if named_action == ACTION_BUILD_SUPPLY_DEPOT:
            action, self.last_worker, self.move_number = build_structure_raw_pt(obs, units.Terran.SupplyDepot, sc2._BUILD_SUPPLY_DEPOT, self.move_number,self.last_worker, self.base_top_left, max_amount = 8)
            return action

        # BUILD REFINERY
        if named_action == ACTION_BUILD_REFINERY:
            action, self.last_worker, self.move_number = build_gas_structure_raw_unit(obs, units.Terran.Refinery, sc2._BUILD_REFINERY, sc2_env.Race.terran, self.move_number, self.last_worker)        
            return action

        # BUILD ENGINEERINGBAY
        if named_action == ACTION_BUILD_ENGINEERINGBAY:
            action, self.last_worker, self.move_number = build_structure_raw_pt(obs, units.Terran.EngineeringBay, sc2._BUILD_ENGINEERINGBAY, self.move_number, self.last_worker, self.base_top_left, max_amount = 1)
            return action

        # BUILD ARMORY
        if named_action == ACTION_BUILD_ARMORY:
            action, self.last_worker, self.move_number = build_structure_raw_pt(obs, units.Terran.Armory, sc2._BUILD_ARMORY, self.move_number, self.last_worker, self.base_top_left, max_amount = 1)
            return action

        # BUILD MISSILE TURRET
        if named_action == ACTION_BUILD_MISSILETURRET:
            action, self.last_worker, self.move_number = build_structure_raw_pt(obs, units.Terran.MissileTurret, sc2._BUILD_MISSILETURRET, self.move_number, self.last_worker, self.base_top_left, max_amount = 8)
            return action

        # BUILD SENSOR TOWER
        if named_action == ACTION_BUILD_SENSORTOWER:
            action, self.last_worker, self.move_number = build_structure_raw_pt(obs, units.Terran.SensorTower, sc2._BUILD_SENSORTOWER, self.move_number, self.last_worker, self.base_top_left, max_amount = 3)
            return action

        # BUILD BUNKER
        if named_action == ACTION_BUILD_BUNKER:
            action, self.last_worker, self.move_number = build_structure_raw_pt(obs, units.Terran.Bunker, sc2._BUILD_BUNKER, self.move_number, self.last_worker, self.base_top_left, max_amount = 5)
            return action

        # BUILD FUSIONCORE
        if named_action == ACTION_BUILD_FUSIONCORE:
            action, self.last_worker, self.move_number = build_structure_raw_pt(obs, units.Terran.FusionCore, sc2._BUILD_FUSIONCORE, self.move_number, self.last_worker, self.base_top_left, max_amount = 1)
            return action

        # BUILD GHOSTACADEMY
        if named_action == ACTION_BUILD_GHOSTACADEMY:
            action, self.last_worker, self.move_number = build_structure_raw_pt(obs, units.Terran.GhostAcademy, sc2._BUILD_GHOSTACADEMY, self.move_number, self.last_worker, self.base_top_left, max_amount = 1)
            return action

        # BUILD BARRACKS
        if named_action == ACTION_BUILD_BARRACKS:
            action, self.last_worker, self.move_number = build_structure_raw_pt(obs, units.Terran.Barracks, sc2._BUILD_BARRACKS, self.move_number, self.last_worker, self.base_top_left, max_amount = 3)
            return action

        # BUILD FACTORY
        if named_action == ACTION_BUILD_FACTORY:
            action, self.last_worker, self.move_number = build_structure_raw_pt(obs, units.Terran.Factory, sc2._BUILD_FACTORY, self.move_number, self.last_worker, self.base_top_left, max_amount = 2)
            return action

        # BUILD STARPORT
        if named_action == ACTION_BUILD_STARPORT:
            action, self.last_worker, self.move_number = build_structure_raw_pt(obs, units.Terran.Starport, sc2._BUILD_STARPORT, self.move_number, self.last_worker, self.base_top_left, max_amount = 2)
            return action

        # BUILD TECHLAB BARRACKS
        if named_action == ACTION_BUILD_TECHLAB_BARRACKS:
            action, self.last_worker, self.move_number = build_structure_raw(obs, units.Terran.Barracks, sc2._BUILD_TECHLAB_BARRACKS, self.move_number, self.last_worker)
            return action
            
        # BUILD TECHLAB FACTORY
        if named_action == ACTION_BUILD_TECHLAB_FACTORY:
            action, self.last_worker, self.move_number = build_structure_raw(obs, units.Terran.Factory, sc2._BUILD_TECHLAB_FACTORY, self.move_number, self.last_worker)
            return action

        # BUILD TECHLAB STARPORT
        if named_action == ACTION_BUILD_TECHLAB_STARPORT:
            action, self.last_worker, self.move_number = build_structure_raw(obs, units.Terran.Starport, sc2._BUILD_TECHLAB_STARPORT, self.move_number, self.last_worker)
            return action

        # BUILD REACTOR BARRACKS
        if named_action == ACTION_BUILD_REACTOR_BARRACKS:
            action, self.last_worker, self.move_number = build_structure_raw(obs, units.Terran.Barracks, sc2._BUILD_REACTOR_BARRACKS, self.move_number, self.last_worker)
            return action

        # BUILD REACTOR FACTORY
        if named_action == ACTION_BUILD_REACTOR_FACTORY:
            action, self.last_worker, self.move_number = build_structure_raw(obs, units.Terran.Factory, sc2._BUILD_REACTOR_FACTORY, self.move_number, self.last_worker)
            return action

        # BUILD REACTOR STARPORT
        if named_action == ACTION_BUILD_REACTOR_STARPORT:
            action, self.last_worker, self.move_number = build_structure_raw(obs, units.Terran.Starport, sc2._BUILD_REACTOR_STARPORT, self.move_number, self.last_worker)
            return action
                

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

        '''ENGINEERING BAY RESEARCH'''
        # RESEARCH INFANTRY WEAPONS
        if named_action == ACTION_RESEARCH_INF_WEAPONS:
            return research_upgrade(obs, sc2._RESEARCH_TERRAN_INF_WEAPONS, units.Terran.EngineeringBay)

        # RESEARCH INFANTRY ARMOR
        if named_action == ACTION_RESEARCH_INF_ARMOR:
            return research_upgrade(obs, sc2._RESEARCH_TERRAN_INF_ARMOR, units.Terran.EngineeringBay)

        # RESEARCH HISEC AUTRACKING
        if named_action == ACTION_RESEARCH_HISEC_AUTOTRACKING:
            return research_upgrade(obs, sc2._RESEARCH_TERRAN_HISEC_AUTOTRACKING, units.Terran.EngineeringBay)

        # RESEARCH NEOSTEEL FRAME
        if named_action == ACTION_RESEARCH_NEOSTEEL_FRAME:
            return research_upgrade(obs, sc2._RESEARCH_TERRAN_NEOSTEEL_FRAME, units.Terran.EngineeringBay)

        # RESEARCH STRUCTURE ARMOR
        if named_action == ACTION_RESEARCH_STRUCTURE_ARMOR:
            return research_upgrade(obs, sc2._RESEARCH_TERRAN_STRUCTURE_ARMOR, units.Terran.EngineeringBay)

        '''ARMORY RESEARCH'''
        # RESEARCH SHIPS WEAPONS
        if named_action == ACTION_RESEARCH_SHIPS_WEAPONS:
            return research_upgrade(obs, sc2._RESEARCH_TERRAN_SHIPS_WEAPONS, units.Terran.Armory)

        # RESEARCH VEHIC WEAPONS
        if named_action == ACTION_RESEARCH_VEHIC_WEAPONS:
            return research_upgrade(obs, sc2._RESEARCH_TERRAN_VEHIC_WEAPONS, units.Terran.Armory)

        # RESEARCH SHIPVEHIC PLATES
        if named_action == ACTION_RESEARCH_SHIPVEHIC_PLATES:
            return research_upgrade(obs, sc2._RESEARCH_TERRAN_SHIPVEHIC_PLATES, units.Terran.Armory)

        '''GHOST ACADEMY RESEARCH'''
        if named_action == ACTION_RESEARCH_GHOST_CLOAK:
            return research_upgrade(obs, sc2._RESEARCH_TERRAN_GHOST_CLOAK, units.Terran.GhostAcademy)

        '''BARRACK RESEARCH'''
        # RESEARCH STIMPACK
        if named_action == ACTION_RESEARCH_STIMPACK:
            return research_upgrade(obs, sc2._RESEARCH_TERRAN_STIMPACK, units.Terran.BarracksTechLab)

        # RESEARCH COMBATSHIELD
        if named_action == ACTION_RESEARCH_COMBATSHIELD:
            return research_upgrade(obs, sc2._RESEARCH_TERRAN_COMBATSHIELD, units.Terran.BarracksTechLab)

        # RESEARCH CONCUSSIVESHELL
        if named_action == ACTION_RESEARCH_CONCUSSIVESHELL:
            return research_upgrade(obs, sc2._RESEARCH_TERRAN_CONCUSSIVESHELL, units.Terran.BarracksTechLab)

        '''FACTORY RESEARCH'''
        # RESEARCH INFERNAL PREIGNITER
        if named_action == ACTION_RESEARCH_INFERNAL_PREIGNITER:
            return research_upgrade(obs, sc2._RESEARCH_TERRAN_INFERNAL_PREIGNITER, units.Terran.FactoryTechLab)

        # RESEARCH DRILLING CLAWS
        if named_action == ACTION_RESEARCH_DRILLING_CLAWS:
            return research_upgrade(obs, sc2._RESEARCH_TERRAN_DRILLING_CLAWS, units.Terran.FactoryTechLab)
        
        # RESEARCH CYCLONE LOCK ON DMG
        if named_action == ACTION_RESEARCH_CYCLONE_LOCKONDMG:
            return research_upgrade(obs, sc2._RESEARCH_TERRAN_CYCLONE_LOCKONDMG, units.Terran.FactoryTechLab)

        # RESEARCH CYCLONE RAPID FIRE
        if named_action == ACTION_RESEARCH_CYCLONE_RAPIDFIRE:
            return research_upgrade(obs, sc2._RESEARCH_TERRAN_CYCLONE_RAPIDFIRE, units.Terran.FactoryTechLab)

        '''STARPORT RESEARCH'''
        # RESEARCH HIGH CAPACITY FUEL
        if named_action == ACTION_RESEARCH_HIGHCAPACITYFUEL:
            return research_upgrade(obs, sc2._RESEARCH_TERRAN_HIGHCAPACITYFUEL, units.Terran.StarportTechLab)
        
        # RESEARCH CORVID REACTOR
        if named_action == ACTION_RESEARCH_CORVIDREACTOR:
            return research_upgrade(obs, sc2._RESEARCH_TERRAN_CORVIDREACTOR, units.Terran.StarportTechLab)

        # RESEARCH BANSHEE CLOAK
        if named_action == ACTION_RESEARCH_BANSHEECLOAK:
            return research_upgrade(obs, sc2._RESEARCH_TERRAN_BANSHEECLOAK, units.Terran.StarportTechLab)

        # RESEARCH BANSHEE HYPERFLIGHT
        if named_action == ACTION_RESEARCH_BANSHEEHYPERFLIGHT:
            return research_upgrade(obs, sc2._RESEARCH_TERRAN_BANSHEEHYPERFLIGHT, units.Terran.StarportTechLab)

        # RESEARCH ADVANCED BALLISTICS
        if named_action == ACTION_RESEARCH_ADVANCEDBALLISTICS:
            return research_upgrade(obs, sc2._RESEARCH_TERRAN_ADVANCEDBALLISTICS, units.Terran.StarportTechLab)



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

        '''BARRACKS UNITS'''
        # TRAIN MARINE
        if named_action == ACTION_TRAIN_MARINE:
            return train_unit(obs, sc2._TRAIN_MARINE, units.Terran.Barracks)

        # TRAIN MARAUDER
        if named_action == ACTION_TRAIN_MARAUDER:
            return train_unit(obs, sc2._TRAIN_MARAUDER, units.Terran.Barracks)

        # TRAIN REAPER
        if named_action == ACTION_TRAIN_REAPER:
            return train_unit(obs, sc2._TRAIN_REAPER, units.Terran.Barracks)

        # TRAIN GHOST
        if named_action == ACTION_TRAIN_GHOST:
            return train_unit(obs, sc2._TRAIN_GHOST, units.Terran.Barracks)
        
        '''FACTORY UNITS'''
        # TRAIN HELLION
        if named_action == ACTION_TRAIN_HELLION:
            return train_unit(obs, sc2._TRAIN_HELLION, units.Terran.Factory)

        # TRAIN HELLBAT
        if named_action == ACTION_TRAIN_HELLBAT:
            return train_unit(obs, sc2._TRAIN_HELLBAT, units.Terran.Factory)

        # TRAIN SIEGETANK
        if named_action == ACTION_TRAIN_SIEGETANK:
            return train_unit(obs, sc2._TRAIN_SIEGETANK, units.Terran.Factory)

        # TRAIN CYCLONE
        if named_action == ACTION_TRAIN_CYCLONE:
            return train_unit(obs, sc2._TRAIN_CYCLONE, units.Terran.Factory)

        # TRAIN WIDOWMINE
        if named_action == ACTION_TRAIN_WIDOWMINE:
            return train_unit(obs, sc2._TRAIN_WIDOWMINE, units.Terran.Factory)
        
        # TRAIN THOR
        if named_action == ACTION_TRAIN_THOR:
            return train_unit(obs, sc2._TRAIN_THOR, units.Terran.Factory)

        '''STARPORT UNITS'''
        # TRAIN VIKING
        if named_action == ACTION_TRAIN_VIKING:
            return train_unit(obs, sc2._TRAIN_VIKING, units.Terran.Starport)

        # TRAIN MEDIVAC
        if named_action == ACTION_TRAIN_MEDIVAC:
            return train_unit(obs, sc2._TRAIN_MEDIVAC, units.Terran.Starport)
        
        # TRAIN LIBERATOR
        if named_action == ACTION_TRAIN_LIBERATOR:
            return train_unit(obs, sc2._TRAIN_LIBERATOR, units.Terran.Starport)

        # TRAIN RAVEN
        if named_action == ACTION_TRAIN_RAVEN:
            return train_unit(obs, sc2._TRAIN_RAVEN, units.Terran.Starport)

        # TRAIN BANSHEE
        if named_action == ACTION_TRAIN_BANSHEE:
            return train_unit(obs, sc2._TRAIN_BANSHEE, units.Terran.Starport)

        # TRAIN BATTLECRUISER
        if named_action == ACTION_TRAIN_BATTLECRUISER:
            return train_unit(obs, sc2._TRAIN_BATTLECRUISER, units.Terran.Starport)
                
        

        # ATTACK ENEMY BASE
        if named_action == ACTION_ATTACK_ENEMY_BASE:
            attack_xy = (38, 44) if self.base_top_left else (19, 23)
            x_offset = random.randint(-4, 4)
            y_offset = random.randint(-4, 4)
            if self.units_to_attack == sc2._NO_UNITS:
                army = select_army(obs, sc2_env.Race.terran)
            else:
                army = self.units_to_attack
            if army != sc2._NO_UNITS:
                target = [attack_xy[0] + x_offset, attack_xy[1] + y_offset]
                action, self.units_to_attack = attack_target_point(obs, army, target)
                self.last_attack_action = ACTION_ATTACK_ENEMY_BASE
                return action
            return no_op()

        # ATTACK ENEMY SECOND BASE
        if named_action == ACTION_ATTACK_ENEMY_SECOND_BASE:
            attack_xy = (19, 44) if self.base_top_left else (38, 23)
            x_offset = random.randint(-4, 4)
            y_offset = random.randint(-4, 4)
            if self.units_to_attack == sc2._NO_UNITS:
                army = select_army(obs, sc2_env.Race.terran)
            else:
                army = self.units_to_attack
            if army != sc2._NO_UNITS:
                target = [attack_xy[0] + x_offset, attack_xy[1] + y_offset]
                action, self.units_to_attack = attack_target_point(obs, army, target)
                self.last_attack_action = ACTION_ATTACK_ENEMY_SECOND_BASE
                return action
            return no_op()

        # ATTACK MY BASE
        if named_action == ACTION_ATTACK_MY_BASE:
            attack_xy = (19, 23) if self.base_top_left else (38, 44)
            x_offset = random.randint(-4, 4)
            y_offset = random.randint(-4, 4)
            if self.units_to_attack == sc2._NO_UNITS:
                army = select_army(obs, sc2_env.Race.terran)
            else:
                army = self.units_to_attack
            if army != sc2._NO_UNITS:
                target = [attack_xy[0] + x_offset, attack_xy[1] + y_offset]
                action, self.units_to_attack = attack_target_point(obs, army, target)
                self.last_attack_action = ACTION_ATTACK_MY_BASE
                return action
            return no_op()

        # ATTACK MY SECOND BASE
        if named_action == ACTION_ATTACK_MY_SECOND_BASE:
            attack_xy = (38, 23) if self.base_top_left else (19, 44)
            x_offset = random.randint(-4, 4)
            y_offset = random.randint(-4, 4)
            if self.units_to_attack == sc2._NO_UNITS:
                army = select_army(obs, sc2_env.Race.terran)
            else:
                army = self.units_to_attack
            if army != sc2._NO_UNITS:
                target = [attack_xy[0] + x_offset, attack_xy[1] + y_offset]
                action, self.units_to_attack = attack_target_point(obs, army, target)
                self.last_attack_action = ACTION_ATTACK_MY_SECOND_BASE
                return action
            return no_op()


        return no_op()