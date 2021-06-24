import random
import numpy as np
from agents.actions.sc2_wrapper import SC2Wrapper, TerranWrapper
import agents.actions.sc2_wrapper as sc2_wrapper
from agents.actions.sc2 import *
import agents.actions.sc2 as sc2     # importing our action set file so that we can use its constants

from pysc2.lib import features, units
from pysc2.env import sc2_env

class MOspatialTerranWrapper(SC2Wrapper):
    def __init__(self, x_gridsize, y_gridsize, map_size):
        SC2Wrapper.__init__(self)

        self.x_gridsize = x_gridsize
        self.y_gridsize = y_gridsize
        self.map_size = map_size

        self.named_actions = [
            sc2_wrapper.ACTION_DO_NOTHING,

            sc2_wrapper.ACTION_BUILD_COMMAND_CENTER,
            sc2_wrapper.ACTION_BUILD_SUPPLY_DEPOT,
            sc2_wrapper.ACTION_BUILD_REFINERY,
            # sc2_wrapper.ACTION_BUILD_ENGINEERINGBAY,
            # sc2_wrapper.ACTION_BUILD_ARMORY,
            # sc2_wrapper.ACTION_BUILD_MISSILETURRET,
            # ACTION_BUILD_SENSORTOWER,
            # ACTION_BUILD_BUNKER,
            # ACTION_BUILD_FUSIONCORE,
            # ACTION_BUILD_GHOSTACADEMY,
            sc2_wrapper.ACTION_BUILD_BARRACKS,
            sc2_wrapper.ACTION_BUILD_FACTORY,
            sc2_wrapper.ACTION_BUILD_STARPORT,

            sc2_wrapper.ACTION_BUILD_TECHLAB_BARRACKS,
            sc2_wrapper.ACTION_BUILD_TECHLAB_FACTORY,
            sc2_wrapper.ACTION_BUILD_TECHLAB_STARPORT,
            sc2_wrapper.ACTION_BUILD_REACTOR_BARRACKS,
            sc2_wrapper.ACTION_BUILD_REACTOR_FACTORY,
            sc2_wrapper.ACTION_BUILD_REACTOR_STARPORT,

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
            # ACTION_RESEARCH_STIMPACK,
            # ACTION_RESEARCH_COMBATSHIELD,
            # ACTION_RESEARCH_CONCUSSIVESHELL,

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

            #sc2_wrapper.ACTION_EFFECT_STIMPACK,

            sc2_wrapper.ACTION_TRAIN_SCV,

            sc2_wrapper.ACTION_TRAIN_MARINE,
            sc2_wrapper.ACTION_TRAIN_MARAUDER,
            sc2_wrapper.ACTION_TRAIN_REAPER,
            # ACTION_TRAIN_GHOST,

            sc2_wrapper.ACTION_TRAIN_HELLION,
            sc2_wrapper.ACTION_TRAIN_HELLBAT,
            sc2_wrapper.ACTION_TRAIN_SIEGETANK,
            sc2_wrapper.ACTION_TRAIN_CYCLONE,
            # ACTION_TRAIN_WIDOWMINE,
            # ACTION_TRAIN_THOR,

            sc2_wrapper.ACTION_TRAIN_VIKING,
            sc2_wrapper.ACTION_TRAIN_MEDIVAC,
            sc2_wrapper.ACTION_TRAIN_LIBERATOR,
            sc2_wrapper.ACTION_TRAIN_RAVEN,
            sc2_wrapper.ACTION_TRAIN_BANSHEE,
            # ACTION_TRAIN_BATTLECRUISER,

            sc2_wrapper.ACTION_HARVEST_MINERALS_IDLE,
            sc2_wrapper.ACTION_HARVEST_MINERALS_FROM_GAS,
            sc2_wrapper.ACTION_HARVEST_GAS_FROM_MINERALS,

            sc2_wrapper.ACTION_ATTACK_POINT,
            sc2_wrapper.ACTION_MOVE_TROOPS_POINT
        ]
        self.n_actions_len = len(self.named_actions)


        self.multi_output_ranges = [0, self.n_actions_len, self.n_actions_len+self.x_gridsize, self.n_actions_len+self.x_gridsize+self.y_gridsize]

        self.action_indices = [idx for idx in range(len(self.named_actions))]

    def get_excluded_actions(self, obs):
        excluded_actions = []
        return excluded_actions

    def get_actions(self):
        x_grid_actions = np.arange(self.multi_output_ranges[1], self.multi_output_ranges[1] + self.x_gridsize)
        y_grid_actions = np.arange(self.multi_output_ranges[2], self.multi_output_ranges[2] + self.y_gridsize)
        total_actions = []
        total_actions.extend(self.action_indices)
        total_actions.extend(x_grid_actions)
        total_actions.extend(y_grid_actions)
        return total_actions

    def get_action(self, action_idx, obs):
        action_id, x, y = action_idx

        adjusted_x = x - self.multi_output_ranges[1]
        adjusted_y = y - self.multi_output_ranges[2]

        gridwidth = self.map_size.x/self.x_gridsize
        gridheight = self.map_size.y/self.y_gridsize

        xtarget = int((adjusted_x*gridwidth) + random.uniform(0, gridwidth))
        ytarget = int(adjusted_y*gridheight + random.uniform(0, gridheight))

        target = [xtarget, ytarget]

        named_action = self.named_actions[action_id]
        #named_action, x, y = self.split_action(named_action)

        if len(self.actions_queue) > 0:
            return self.actions_queue.pop(0)

        if self.units_to_attack != sc2._NO_UNITS:
            named_action = self.last_attack_action

        if self.units_to_effect != sc2._NO_UNITS:
            named_action = self.last_effect_action

        if obs.game_loop[0] < 80 and self.base_top_left == None:
            command_center = get_my_units_by_type(obs, units.Terran.CommandCenter)[0]
            self.base_top_left = (command_center.x < 32)


        '''LIST OF ACTIONS THE AGENT IS ABLE TO CHOOSE FROM:'''

        # BUILD COMMAND CENTER
        if named_action == sc2_wrapper.ACTION_BUILD_COMMAND_CENTER:
            actions = build_structure_raw_pt_spatial(obs, units.Terran.CommandCenter, sc2._BUILD_COMMAND_CENTER, target)
            action, self.actions_queue = organize_queue(actions, self.actions_queue)
            return action

        # BUILD SUPPLY DEPOT
        if named_action == sc2_wrapper.ACTION_BUILD_SUPPLY_DEPOT:
            actions = build_structure_raw_pt_spatial(obs, units.Terran.SupplyDepot, sc2._BUILD_SUPPLY_DEPOT, target)
            action, self.actions_queue = organize_queue(actions, self.actions_queue)
            return action

        # BUILD REFINERY
        if named_action == sc2_wrapper.ACTION_BUILD_REFINERY:
            actions = build_gas_structure_raw_unit(obs, units.Terran.Refinery, sc2._BUILD_REFINERY, sc2_env.Race.terran)
            action, self.actions_queue = organize_queue(actions, self.actions_queue)     
            return action

        # BUILD ENGINEERINGBAY
        if named_action == sc2_wrapper.ACTION_BUILD_ENGINEERINGBAY:
            actions = build_structure_raw_pt_spatial(obs, units.Terran.EngineeringBay, sc2._BUILD_ENGINEERINGBAY, target)
            action, self.actions_queue = organize_queue(actions, self.actions_queue)
            return action

        # BUILD ARMORY
        if named_action == sc2_wrapper.ACTION_BUILD_ARMORY:
            actions = build_structure_raw_pt_spatial(obs, units.Terran.Armory, sc2._BUILD_ARMORY, target)
            action, self.actions_queue = organize_queue(actions, self.actions_queue)
            return action

        # BUILD MISSILE TURRET
        if named_action == sc2_wrapper.ACTION_BUILD_MISSILETURRET:
            actions = build_structure_raw_pt_spatial(obs, units.Terran.MissileTurret, sc2._BUILD_MISSILETURRET, target)
            action, self.actions_queue = organize_queue(actions, self.actions_queue)
            return action

        # BUILD SENSOR TOWER
        if named_action == sc2_wrapper.ACTION_BUILD_SENSORTOWER:
            actions = build_structure_raw_pt_spatial(obs, units.Terran.SensorTower, sc2._BUILD_SENSORTOWER, target)
            action, self.actions_queue = organize_queue(actions, self.actions_queue)
            return action

        # BUILD BUNKER
        if named_action == sc2_wrapper.ACTION_BUILD_BUNKER:
            actions = build_structure_raw_pt_spatial(obs, units.Terran.Bunker, sc2._BUILD_BUNKER, target)
            action, self.actions_queue = organize_queue(actions, self.actions_queue)
            return action

        # BUILD FUSIONCORE
        if named_action == sc2_wrapper.ACTION_BUILD_FUSIONCORE:
            actions = build_structure_raw_pt_spatial(obs, units.Terran.FusionCore, sc2._BUILD_FUSIONCORE, target)
            action, self.actions_queue = organize_queue(actions, self.actions_queue)
            return action

        # BUILD GHOSTACADEMY
        if named_action == sc2_wrapper.ACTION_BUILD_GHOSTACADEMY:
            actions = build_structure_raw_pt_spatial(obs, units.Terran.GhostAcademy, sc2._BUILD_GHOSTACADEMY, target)
            action, self.actions_queue = organize_queue(actions, self.actions_queue)
            return action

        # BUILD BARRACKS
        if named_action == sc2_wrapper.ACTION_BUILD_BARRACKS:
            actions = build_structure_raw_pt_spatial(obs, units.Terran.Barracks, sc2._BUILD_BARRACKS, target)
            action, self.actions_queue = organize_queue(actions, self.actions_queue)
            return action

        # BUILD FACTORY
        if named_action == sc2_wrapper.ACTION_BUILD_FACTORY:
            actions = build_structure_raw_pt_spatial(obs, units.Terran.Factory, sc2._BUILD_FACTORY, target)
            action, self.actions_queue = organize_queue(actions, self.actions_queue)
            return action

        # BUILD STARPORT
        if named_action == sc2_wrapper.ACTION_BUILD_STARPORT:
            actions = build_structure_raw_pt_spatial(obs, units.Terran.Starport, sc2._BUILD_STARPORT, target)
            action, self.actions_queue = organize_queue(actions, self.actions_queue)
            return action

        # BUILD TECHLAB BARRACKS
        if named_action == sc2_wrapper.ACTION_BUILD_TECHLAB_BARRACKS:
            actions = build_structure_raw(obs, units.Terran.Barracks, sc2._BUILD_TECHLAB_BARRACKS)
            action, self.actions_queue = organize_queue(actions, self.actions_queue)
            return action
            
        # BUILD TECHLAB FACTORY
        if named_action == sc2_wrapper.ACTION_BUILD_TECHLAB_FACTORY:
            actions = build_structure_raw(obs, units.Terran.Factory, sc2._BUILD_TECHLAB_FACTORY)
            action, self.actions_queue = organize_queue(actions, self.actions_queue)
            return action

        # BUILD TECHLAB STARPORT
        if named_action == sc2_wrapper.ACTION_BUILD_TECHLAB_STARPORT:
            actions = build_structure_raw(obs, units.Terran.Starport, sc2._BUILD_TECHLAB_STARPORT)
            action, self.actions_queue = organize_queue(actions, self.actions_queue)
            return action

        # BUILD REACTOR BARRACKS
        if named_action == sc2_wrapper.ACTION_BUILD_REACTOR_BARRACKS:
            actions = build_structure_raw(obs, units.Terran.Barracks, sc2._BUILD_REACTOR_BARRACKS)
            action, self.actions_queue = organize_queue(actions, self.actions_queue)
            return action

        # BUILD REACTOR FACTORY
        if named_action == sc2_wrapper.ACTION_BUILD_REACTOR_FACTORY:
            actions = build_structure_raw(obs, units.Terran.Factory, sc2._BUILD_REACTOR_FACTORY)
            action, self.actions_queue = organize_queue(actions, self.actions_queue)
            return action

        # BUILD REACTOR STARPORT
        if named_action == sc2_wrapper.ACTION_BUILD_REACTOR_STARPORT:
            actions = build_structure_raw(obs, units.Terran.Starport, sc2._BUILD_REACTOR_STARPORT)
            action, self.actions_queue = organize_queue(actions, self.actions_queue)
            return action
                

        # HARVEST MINERALS WITH IDLE WORKER
        if named_action == sc2_wrapper.ACTION_HARVEST_MINERALS_IDLE:
            idle_workers = get_all_idle_workers(obs, sc2_env.Race.terran)
            if idle_workers != sc2._NO_UNITS:
                return harvest_gather_minerals_idle(obs, sc2_env.Race.terran, idle_workers)
            return no_op()
            
        # TO DO: Create a harvest minerals with worker from refinery line so the bot can juggle workers from mineral lines to gas back and forth

        # HARVEST MINERALS WITH WORKER FROM GAS LINE
        if named_action == sc2_wrapper.ACTION_HARVEST_MINERALS_FROM_GAS:
            if building_exists(obs, units.Terran.CommandCenter) or building_exists(obs, units.Terran.PlanetaryFortress) or building_exists(obs, units.Terran.OrbitalCommand):
                return harvest_gather_minerals(obs, sc2_env.Race.terran)
            return no_op()

        # HARVEST GAS WITH WORKER FROM MINERAL LINE
        if named_action == sc2_wrapper.ACTION_HARVEST_GAS_FROM_MINERALS:
            if building_exists(obs, units.Terran.Refinery):
                return harvest_gather_gas(obs, sc2_env.Race.terran)
            return no_op()

        '''ENGINEERING BAY RESEARCH'''
        # RESEARCH INFANTRY WEAPONS
        if named_action == sc2_wrapper.ACTION_RESEARCH_INF_WEAPONS:
            return research_upgrade(obs, sc2._RESEARCH_TERRAN_INF_WEAPONS, units.Terran.EngineeringBay)

        # RESEARCH INFANTRY ARMOR
        if named_action == sc2_wrapper.ACTION_RESEARCH_INF_ARMOR:
            return research_upgrade(obs, sc2._RESEARCH_TERRAN_INF_ARMOR, units.Terran.EngineeringBay)

        # RESEARCH HISEC AUTRACKING
        if named_action == sc2_wrapper.ACTION_RESEARCH_HISEC_AUTOTRACKING:
            return research_upgrade(obs, sc2._RESEARCH_TERRAN_HISEC_AUTOTRACKING, units.Terran.EngineeringBay)

        # RESEARCH NEOSTEEL FRAME
        if named_action == sc2_wrapper.ACTION_RESEARCH_NEOSTEEL_FRAME:
            return research_upgrade(obs, sc2._RESEARCH_TERRAN_NEOSTEEL_FRAME, units.Terran.EngineeringBay)

        # RESEARCH STRUCTURE ARMOR
        if named_action == sc2_wrapper.ACTION_RESEARCH_STRUCTURE_ARMOR:
            return research_upgrade(obs, sc2._RESEARCH_TERRAN_STRUCTURE_ARMOR, units.Terran.EngineeringBay)

        '''ARMORY RESEARCH'''
        # RESEARCH SHIPS WEAPONS
        if named_action == sc2_wrapper.ACTION_RESEARCH_SHIPS_WEAPONS:
            return research_upgrade(obs, sc2._RESEARCH_TERRAN_SHIPS_WEAPONS, units.Terran.Armory)

        # RESEARCH VEHIC WEAPONS
        if named_action == sc2_wrapper.ACTION_RESEARCH_VEHIC_WEAPONS:
            return research_upgrade(obs, sc2._RESEARCH_TERRAN_VEHIC_WEAPONS, units.Terran.Armory)

        # RESEARCH SHIPVEHIC PLATES
        if named_action == sc2_wrapper.ACTION_RESEARCH_SHIPVEHIC_PLATES:
            return research_upgrade(obs, sc2._RESEARCH_TERRAN_SHIPVEHIC_PLATES, units.Terran.Armory)

        '''GHOST ACADEMY RESEARCH'''
        if named_action == sc2_wrapper.ACTION_RESEARCH_GHOST_CLOAK:
            return research_upgrade(obs, sc2._RESEARCH_TERRAN_GHOST_CLOAK, units.Terran.GhostAcademy)

        '''BARRACK RESEARCH'''
        # RESEARCH STIMPACK
        if named_action == sc2_wrapper.ACTION_RESEARCH_STIMPACK:
            return research_upgrade(obs, sc2._RESEARCH_TERRAN_STIMPACK, units.Terran.BarracksTechLab)

        # RESEARCH COMBATSHIELD
        if named_action == sc2_wrapper.ACTION_RESEARCH_COMBATSHIELD:
            return research_upgrade(obs, sc2._RESEARCH_TERRAN_COMBATSHIELD, units.Terran.BarracksTechLab)

        # RESEARCH CONCUSSIVESHELL
        if named_action == sc2_wrapper.ACTION_RESEARCH_CONCUSSIVESHELL:
            return research_upgrade(obs, sc2._RESEARCH_TERRAN_CONCUSSIVESHELL, units.Terran.BarracksTechLab)

        '''FACTORY RESEARCH'''
        # RESEARCH INFERNAL PREIGNITER
        if named_action == sc2_wrapper.ACTION_RESEARCH_INFERNAL_PREIGNITER:
            return research_upgrade(obs, sc2._RESEARCH_TERRAN_INFERNAL_PREIGNITER, units.Terran.FactoryTechLab)

        # RESEARCH DRILLING CLAWS
        if named_action == sc2_wrapper.ACTION_RESEARCH_DRILLING_CLAWS:
            return research_upgrade(obs, sc2._RESEARCH_TERRAN_DRILLING_CLAWS, units.Terran.FactoryTechLab)
        
        # RESEARCH CYCLONE LOCK ON DMG
        if named_action == sc2_wrapper.ACTION_RESEARCH_CYCLONE_LOCKONDMG:
            return research_upgrade(obs, sc2._RESEARCH_TERRAN_CYCLONE_LOCKONDMG, units.Terran.FactoryTechLab)

        # RESEARCH CYCLONE RAPID FIRE
        if named_action == sc2_wrapper.ACTION_RESEARCH_CYCLONE_RAPIDFIRE:
            return research_upgrade(obs, sc2._RESEARCH_TERRAN_CYCLONE_RAPIDFIRE, units.Terran.FactoryTechLab)

        '''STARPORT RESEARCH'''
        # RESEARCH HIGH CAPACITY FUEL
        if named_action == sc2_wrapper.ACTION_RESEARCH_HIGHCAPACITYFUEL:
            return research_upgrade(obs, sc2._RESEARCH_TERRAN_HIGHCAPACITYFUEL, units.Terran.StarportTechLab)
        
        # RESEARCH CORVID REACTOR
        if named_action == sc2_wrapper.ACTION_RESEARCH_CORVIDREACTOR:
            return research_upgrade(obs, sc2._RESEARCH_TERRAN_CORVIDREACTOR, units.Terran.StarportTechLab)

        # RESEARCH BANSHEE CLOAK
        if named_action == sc2_wrapper.ACTION_RESEARCH_BANSHEECLOAK:
            return research_upgrade(obs, sc2._RESEARCH_TERRAN_BANSHEECLOAK, units.Terran.StarportTechLab)

        # RESEARCH BANSHEE HYPERFLIGHT
        if named_action == sc2_wrapper.ACTION_RESEARCH_BANSHEEHYPERFLIGHT:
            return research_upgrade(obs, sc2._RESEARCH_TERRAN_BANSHEEHYPERFLIGHT, units.Terran.StarportTechLab)

        # RESEARCH ADVANCED BALLISTICS
        if named_action == sc2_wrapper.ACTION_RESEARCH_ADVANCEDBALLISTICS:
            return research_upgrade(obs, sc2._RESEARCH_TERRAN_ADVANCEDBALLISTICS, units.Terran.StarportTechLab)


        # TRAIN SCV
        if named_action == sc2_wrapper.ACTION_TRAIN_SCV:
            return train_unit(obs, sc2._TRAIN_SCV, units.Terran.CommandCenter)

        '''BARRACKS UNITS'''
        # TRAIN MARINE
        if named_action == sc2_wrapper.ACTION_TRAIN_MARINE:
            return train_unit(obs, sc2._TRAIN_MARINE, units.Terran.Barracks)

        # TRAIN MARAUDER
        if named_action == sc2_wrapper.ACTION_TRAIN_MARAUDER:
            return train_unit(obs, sc2._TRAIN_MARAUDER, units.Terran.Barracks)

        # TRAIN REAPER
        if named_action == sc2_wrapper.ACTION_TRAIN_REAPER:
            return train_unit(obs, sc2._TRAIN_REAPER, units.Terran.Barracks)

        # TRAIN GHOST
        if named_action == sc2_wrapper.ACTION_TRAIN_GHOST:
            return train_unit(obs, sc2._TRAIN_GHOST, units.Terran.Barracks)
        
        '''FACTORY UNITS'''
        # TRAIN HELLION
        if named_action == sc2_wrapper.ACTION_TRAIN_HELLION:
            return train_unit(obs, sc2._TRAIN_HELLION, units.Terran.Factory)

        # TRAIN HELLBAT
        if named_action == sc2_wrapper.ACTION_TRAIN_HELLBAT:
            return train_unit(obs, sc2._TRAIN_HELLBAT, units.Terran.Factory)

        # TRAIN SIEGETANK
        if named_action == sc2_wrapper.ACTION_TRAIN_SIEGETANK:
            return train_unit(obs, sc2._TRAIN_SIEGETANK, units.Terran.Factory)

        # TRAIN CYCLONE
        if named_action == sc2_wrapper.ACTION_TRAIN_CYCLONE:
            return train_unit(obs, sc2._TRAIN_CYCLONE, units.Terran.Factory)

        # TRAIN WIDOWMINE
        if named_action == sc2_wrapper.ACTION_TRAIN_WIDOWMINE:
            return train_unit(obs, sc2._TRAIN_WIDOWMINE, units.Terran.Factory)
        
        # TRAIN THOR
        if named_action == sc2_wrapper.ACTION_TRAIN_THOR:
            return train_unit(obs, sc2._TRAIN_THOR, units.Terran.Factory)

        '''STARPORT UNITS'''
        # TRAIN VIKING
        if named_action == sc2_wrapper.ACTION_TRAIN_VIKING:
            return train_unit(obs, sc2._TRAIN_VIKING, units.Terran.Starport)

        # TRAIN MEDIVAC
        if named_action == sc2_wrapper.ACTION_TRAIN_MEDIVAC:
            return train_unit(obs, sc2._TRAIN_MEDIVAC, units.Terran.Starport)
        
        # TRAIN LIBERATOR
        if named_action == sc2_wrapper.ACTION_TRAIN_LIBERATOR:
            return train_unit(obs, sc2._TRAIN_LIBERATOR, units.Terran.Starport)

        # TRAIN RAVEN
        if named_action == sc2_wrapper.ACTION_TRAIN_RAVEN:
            return train_unit(obs, sc2._TRAIN_RAVEN, units.Terran.Starport)

        # TRAIN BANSHEE
        if named_action == sc2_wrapper.ACTION_TRAIN_BANSHEE:
            return train_unit(obs, sc2._TRAIN_BANSHEE, units.Terran.Starport)

        # TRAIN BATTLECRUISER
        if named_action == sc2_wrapper.ACTION_TRAIN_BATTLECRUISER:
            return train_unit(obs, sc2._TRAIN_BATTLECRUISER, units.Terran.Starport)
                
        
        # EFFECT STIMPACK
        if named_action == sc2_wrapper.ACTION_EFFECT_STIMPACK:
            if self.units_to_effect == sc2._NO_UNITS:
                army = []
                marines = get_my_units_by_type(obs, units.Terran.Marine)
                marauders = get_my_units_by_type(obs, units.Terran.Marauder)
                army.extend(marines)
                army.extend(marauders)
                if len(army) == 0:
                    army = sc2._NO_UNITS
            else:
                army = self.units_to_effect

            if army != sc2._NO_UNITS:
                action, self.units_to_effect = effect_units(obs, sc2._EFFECT_STIMPACK, army)
                self.last_effect_action = sc2_wrapper.ACTION_EFFECT_STIMPACK
                return action
            return no_op()

        # ATTACK ACTION
        if named_action == sc2_wrapper.ACTION_ATTACK_POINT:
            actions = attack_target_point_spatial(obs, sc2_env.Race.terran, target)
            action, self.actions_queue = organize_queue(actions, self.actions_queue)
            return action

        # MOVE TROOPS
        if named_action == sc2_wrapper.ACTION_MOVE_TROOPS_POINT:
            return no_op()

        # ATTACK MY BASE
        # if named_action == ACTION_ATTACK_MY_BASE:
        #     target=self.my_base_xy
        #     actions = attack_target_point(obs, sc2_env.Race.terran, target, self.base_top_left)
        #     action, self.actions_queue = organize_queue(actions, self.actions_queue)
        #     return action

        # # ATTACK MY SECOND BASE
        # if named_action == ACTION_ATTACK_MY_SECOND_BASE:
        #     target=self.my_second_base_xy
        #     actions = attack_target_point(obs, sc2_env.Race.terran, target, self.base_top_left)
        #     action, self.actions_queue = organize_queue(actions, self.actions_queue)
        #     return action

        # # # ATTACK ENEMY BASE
        # if named_action == ACTION_ATTACK_ENEMY_BASE:
        #     target=self.enemy_base_xy
        #     actions = attack_target_point(obs, sc2_env.Race.terran, target, self.base_top_left)
        #     action, self.actions_queue = organize_queue(actions, self.actions_queue)
        #     return action

        # # # ATTACK ENEMY SECOND BASE
        # if named_action == ACTION_ATTACK_ENEMY_SECOND_BASE:
        #     target=self.enemy_second_base_xy
        #     actions = attack_target_point(obs, sc2_env.Race.terran, target, self.base_top_left)
        #     action, self.actions_queue = organize_queue(actions, self.actions_queue)
        #     return action

        # # ATTACK DISTRIBUTE ARMY
        # if named_action == ACTION_ATTACK_DISTRIBUTE_ARMY:
        #     actions = attack_distribute_army(obs, sc2_env.Race.terran)
        #     action, self.actions_queue = organize_queue(actions, self.actions_queue)
        #     return action

        return no_op()


class SimpleMOTerranWrapper(TerranWrapper):
    def __init__(self, x_gridsize, y_gridsize, map_size):
        SC2Wrapper.__init__(self)

        self.x_gridsize = x_gridsize
        self.y_gridsize = y_gridsize
        self.map_size = map_size

        self.named_actions = [
            sc2_wrapper.ACTION_DO_NOTHING,

            sc2_wrapper.ACTION_BUILD_COMMAND_CENTER,
            sc2_wrapper.ACTION_BUILD_SUPPLY_DEPOT,
            sc2_wrapper.ACTION_BUILD_REFINERY,
            sc2_wrapper.ACTION_BUILD_ENGINEERINGBAY,
            sc2_wrapper.ACTION_BUILD_ARMORY,
            sc2_wrapper.ACTION_BUILD_MISSILETURRET,
            # sc2_wrapper.ACTION_BUILD_SENSORTOWER,
            # sc2_wrapper.ACTION_BUILD_BUNKER,
            sc2_wrapper.ACTION_BUILD_FUSIONCORE,
            # sc2_wrapper.ACTION_BUILD_GHOSTACADEMY,
            sc2_wrapper.ACTION_BUILD_BARRACKS,
            sc2_wrapper.ACTION_BUILD_FACTORY,
            sc2_wrapper.ACTION_BUILD_STARPORT,
            
            sc2_wrapper.ACTION_BUILD_TECHLAB_BARRACKS,
            sc2_wrapper.ACTION_BUILD_TECHLAB_FACTORY,
            sc2_wrapper.ACTION_BUILD_TECHLAB_STARPORT,
            sc2_wrapper.ACTION_BUILD_REACTOR_BARRACKS,
            sc2_wrapper.ACTION_BUILD_REACTOR_FACTORY,
            sc2_wrapper.ACTION_BUILD_REACTOR_STARPORT,

            # # ENGINEERING BAY RESEARCH
            # sc2_wrapper.ACTION_RESEARCH_INF_WEAPONS,
            # sc2_wrapper.ACTION_RESEARCH_INF_ARMOR,
            # sc2_wrapper.ACTION_RESEARCH_HISEC_AUTOTRACKING,
            # sc2_wrapper.ACTION_RESEARCH_NEOSTEEL_FRAME,
            # sc2_wrapper.ACTION_RESEARCH_STRUCTURE_ARMOR,
            
            # # ARMORY RESEARCH
            # sc2_wrapper.ACTION_RESEARCH_SHIPS_WEAPONS,
            # sc2_wrapper.ACTION_RESEARCH_VEHIC_WEAPONS,
            # sc2_wrapper.ACTION_RESEARCH_SHIPVEHIC_PLATES,

            # # GHOST ACADEMY RESEARCH
            # sc2_wrapper.ACTION_RESEARCH_GHOST_CLOAK,

            # # BARRACKS RESEARCH
            sc2_wrapper.ACTION_RESEARCH_STIMPACK,
            # sc2_wrapper.ACTION_RESEARCH_COMBATSHIELD,
            # sc2_wrapper.ACTION_RESEARCH_CONCUSSIVESHELL,

            # # FACTORY RESEARCH
            # sc2_wrapper.ACTION_RESEARCH_INFERNAL_PREIGNITER,
            # sc2_wrapper.ACTION_RESEARCH_DRILLING_CLAWS,
            # sc2_wrapper.ACTION_RESEARCH_CYCLONE_LOCKONDMG,
            # sc2_wrapper.ACTION_RESEARCH_CYCLONE_RAPIDFIRE,

            # # STARPORT RESEARCH
            # sc2_wrapper.ACTION_RESEARCH_HIGHCAPACITYFUEL,
            # sc2_wrapper.ACTION_RESEARCH_CORVIDREACTOR,
            # sc2_wrapper.ACTION_RESEARCH_BANSHEECLOAK,
            # sc2_wrapper.ACTION_RESEARCH_BANSHEEHYPERFLIGHT,
            # sc2_wrapper.ACTION_RESEARCH_ADVANCEDBALLISTICS,

            # # FUSION CORE RESEARCH
            # sc2_wrapper.ACTION_RESEARCH_BATTLECRUISER_WEAPONREFIT,

            sc2_wrapper.ACTION_EFFECT_STIMPACK,

            sc2_wrapper.ACTION_TRAIN_SCV,

            sc2_wrapper.ACTION_TRAIN_MARINE,
            sc2_wrapper.ACTION_TRAIN_MARAUDER,
            sc2_wrapper.ACTION_TRAIN_REAPER,
            #sc2_wrapper.ACTION_TRAIN_GHOST,

            sc2_wrapper.ACTION_TRAIN_HELLION,
            sc2_wrapper.ACTION_TRAIN_HELLBAT,
            sc2_wrapper.ACTION_TRAIN_SIEGETANK,
            sc2_wrapper.ACTION_TRAIN_CYCLONE,
            sc2_wrapper.ACTION_TRAIN_WIDOWMINE,
            sc2_wrapper.ACTION_TRAIN_THOR,

            sc2_wrapper.ACTION_TRAIN_VIKING,
            sc2_wrapper.ACTION_TRAIN_MEDIVAC,
            sc2_wrapper.ACTION_TRAIN_LIBERATOR,
            sc2_wrapper.ACTION_TRAIN_RAVEN,
            sc2_wrapper.ACTION_TRAIN_BANSHEE,
            sc2_wrapper.ACTION_TRAIN_BATTLECRUISER,

            sc2_wrapper.ACTION_HARVEST_MINERALS_IDLE,
            sc2_wrapper.ACTION_HARVEST_MINERALS_FROM_GAS,
            sc2_wrapper.ACTION_HARVEST_GAS_FROM_MINERALS,

            sc2_wrapper.ACTION_ATTACK_POINT,
            sc2_wrapper.ACTION_MOVE_TROOPS_POINT
        ]

        self.n_actions_len = len(self.named_actions)

        self.multi_output_ranges = [0, self.n_actions_len, self.n_actions_len+self.x_gridsize, self.n_actions_len+self.x_gridsize+self.y_gridsize]

        self.action_indices = [idx for idx in range(len(self.named_actions))]

    def get_actions(self):
        x_grid_actions = np.arange(self.multi_output_ranges[1], self.multi_output_ranges[1] + self.x_gridsize)
        y_grid_actions = np.arange(self.multi_output_ranges[2], self.multi_output_ranges[2] + self.y_gridsize)
        total_actions = []
        total_actions.extend(self.action_indices)
        total_actions.extend(x_grid_actions)
        total_actions.extend(y_grid_actions)
        return total_actions

    def get_action(self, action_idx, obs):
        action_id, x, y = action_idx

        adjusted_x = x - self.multi_output_ranges[1]
        adjusted_y = y - self.multi_output_ranges[2]

        gridwidth = self.map_size.x/self.x_gridsize
        gridheight = self.map_size.y/self.y_gridsize

        xtarget = int((adjusted_x*gridwidth) + random.uniform(0, gridwidth))
        ytarget = int(adjusted_y*gridheight + random.uniform(0, gridheight))

        action_idx = [action_id, xtarget, ytarget]
        action = super().get_action(action_idx, obs)
        return action

    def attackpoint(self, obs, x, y):
        target = [x, y]
        actions = attack_target_point_spatial(obs, sc2_env.Race.terran, target)
        action, self.actions_queue = organize_queue(actions, self.actions_queue)
        return action

    