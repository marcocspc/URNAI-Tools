from .defeatenemies import GeneralizedDefeatEnemiesScenario
from urnai.utils.error import EnvironmentNotSupportedError
from urnai.agents.actions.scenarios.rts.generalization.buildunits import BuildUnitsDeepRTSActionWrapper, BuildUnitsStarcraftIIActionWrapper 
from pysc2.lib import actions, features, units
from urnai.agents.actions import sc2 as scaux
from urnai.agents.rewards.default import PureReward
import numpy as np
from pysc2.env import sc2_env
from statistics import mean
import random, math
from urnai.envs.deep_rts import DeepRTSEnv


class GeneralizedBuildUnitsScenario(GeneralizedDefeatEnemiesScenario):

    GAME_DEEP_RTS = "drts" 
    GAME_STARCRAFT_II = "sc2" 
    MAP_1 = "map1"
    MAP_2 = "map2"

    MAP_PLAYER_TOWNHALL_X = 30
    MAP_PLAYER_TOWNHALL_Y = 36
    MAP_PLAYER_BARRACK_X = 39
    MAP_PLAYER_BARRACK_Y = 36
    MAP_PLAYER_FARM_X = 42
    MAP_PLAYER_FARM_Y = 42

    MAP_1_PLAYER_X = 25
    MAP_2_PLAYER_X = 33
    Y_PLAYER_BASE = 33
    MAP_PLAYER_LOCATIONS = {
            MAP_1 : [
                #troops
                {"x" : MAP_1_PLAYER_X, "y" : Y_PLAYER_BASE},
                {"x" : MAP_1_PLAYER_X, "y" : Y_PLAYER_BASE + 1},
                {"x" : MAP_1_PLAYER_X, "y" : Y_PLAYER_BASE + 2},
                {"x" : MAP_1_PLAYER_X, "y" : Y_PLAYER_BASE + 3},
                {"x" : MAP_1_PLAYER_X, "y" : Y_PLAYER_BASE + 4},
                {"x" : MAP_1_PLAYER_X, "y" : Y_PLAYER_BASE + 5},
                {"x" : MAP_1_PLAYER_X + 1, "y" : Y_PLAYER_BASE},
                {"x" : MAP_1_PLAYER_X + 1, "y" : Y_PLAYER_BASE + 1},
                {"x" : MAP_1_PLAYER_X + 1, "y" : Y_PLAYER_BASE + 2},
                {"x" : MAP_1_PLAYER_X + 1, "y" : Y_PLAYER_BASE + 3},
                {"x" : MAP_1_PLAYER_X + 1, "y" : Y_PLAYER_BASE + 4},
                {"x" : MAP_1_PLAYER_X + 1, "y" : Y_PLAYER_BASE + 5},
                ],
            MAP_2 : [
                #troops
                {"x" : MAP_2_PLAYER_X, "y" : Y_PLAYER_BASE},
                {"x" : MAP_2_PLAYER_X, "y" : Y_PLAYER_BASE + 1},
                {"x" : MAP_2_PLAYER_X, "y" : Y_PLAYER_BASE + 2},
                {"x" : MAP_2_PLAYER_X, "y" : Y_PLAYER_BASE + 3},
                {"x" : MAP_2_PLAYER_X, "y" : Y_PLAYER_BASE + 4},
                {"x" : MAP_2_PLAYER_X, "y" : Y_PLAYER_BASE + 5},
                {"x" : MAP_2_PLAYER_X - 1, "y" : Y_PLAYER_BASE},
                {"x" : MAP_2_PLAYER_X - 1, "y" : Y_PLAYER_BASE + 1},
                {"x" : MAP_2_PLAYER_X - 1, "y" : Y_PLAYER_BASE + 2},
                {"x" : MAP_2_PLAYER_X - 1, "y" : Y_PLAYER_BASE + 3},
                {"x" : MAP_2_PLAYER_X - 1, "y" : Y_PLAYER_BASE + 4},
                {"x" : MAP_2_PLAYER_X - 1, "y" : Y_PLAYER_BASE + 5},
                ],
            }

    ACTION_DRTS_COLLECT_GOLD = 17
    ACTION_DRTS_BUILD_FARM = 18
    ACTION_DRTS_BUILD_BARRACK = 19
    ACTION_DRTS_BUILD_FOOTMAN = 20

    def __init__(self, game = GAME_DEEP_RTS, render=False, drts_map="total-64x64-playable-22x16-buildunits.json", sc2_map="BuildMarines", drts_start_oil=999999, drts_start_gold=999999, drts_start_lumber=999999, drts_start_food=999999, fit_to_screen=False):
        super().__init__(game=game, render=render, drts_map=drts_map, sc2_map=sc2_map, drts_number_of_players=1, drts_start_oil=drts_start_oil, drts_start_gold=drts_start_gold, drts_start_lumber=drts_start_lumber, drts_start_food=drts_start_food, fit_to_screen=fit_to_screen)

    def step(self, action):
        if (self.game == GeneralizedBuildUnitsScenario.GAME_DEEP_RTS):
            if self.steps == 0:
                self.setup_map()
                self.spawn_army()

            state, reward, done = None, None, None 
            if action == ACTION_DRTS_COLLECT_GOLD:
                self.collect_gold()
                no_action = 15
                state, reward, done = self.env.step(no_action)
            elif action == ACTION_DRTS_BUILD_FARM:
                self.build_farm()
                no_action = 15
                state, reward, done = self.env.step(no_action)
            elif action == ACTION_DRTS_BUILD_BARRACK:
                self.build_barrack()
                no_action = 15
                state, reward, done = self.env.step(no_action)
            elif action == ACTION_DRTS_BUILD_FOOTMAN:
                self.build_footman()
                no_action = 15
                state, reward, done = self.env.step(no_action)
            else:
                state, reward, done = self.env.step(action)
            self.steps += 1
            return state, reward, done 

        elif (self.game == GeneralizedBuildUnitsScenario.GAME_STARCRAFT_II):
            self.steps += 1
            return self.env.step(action)

    def spawn_army(self):
        for coords in GeneralizedBuildUnitsScenario.MAP_PLAYER_LOCATIONS[self.map_spawn]:
            idx = GeneralizedBuildUnitsScenario.MAP_PLAYER_LOCATIONS[self.map_spawn].index(coords)
            ln = len(GeneralizedBuildUnitsScenario.MAP_PLAYER_LOCATIONS[self.map_spawn]) 
            tile = self.env.game.tilemap.get_tile(coords['x'], coords['y'])
            self.env.game.players[0].spawn_unit(self.env.constants.Unit.Peasant, tile)

        tile = self.env.game.tilemap.get_tile(MAP_PLAYER_TOWNHALL_X, MAP_PLAYER_TOWNHALL_Y)
        self.env.game.players[0].spawn_unit(self.env.constants.Unit.TownHall, tile)

    def collect_gold(self):
        player = 0
        peasants = self.get_player_specific_type_units(player, self.env.constants.Unit.Peasant) 
        gold = 6
        gold_tiles = self.get_tiles_by_id(gold)
        how_many_gold_spots = len(gold_tiles)
        n = how_many_gold_spots
        peasants_sets = [peasants[i * n:(i + 1) * n] for i in range((len(peasants) + n - 1) // n )]

        for i in range(len(peasants_sets)):
            peasant_set = peasants_sets[i]
            gold_tile = gold_tiles[i]

            for peasant in peasant_set:
                peasant.right_click(gold_tile)

    def build_farm(self):
        tile = self.env.game.tilemap.get_tile(MAP_PLAYER_FARM_X, MAP_PLAYER_FARM_Y)
        self.env.game.players[0].spawn_unit(self.env.constants.Unit.Farm, tile)

    def build_barrack(self):
        tile = self.env.game.tilemap.get_tile(MAP_PLAYER_BARRACK_X, MAP_PLAYER_BARRACK_Y)
        self.env.game.players[0].spawn_unit(self.env.constants.Unit.Barracks, tile)

    def build_footman(self):
        player = 0
        barracks_list = self.get_player_specific_type_units(player, self.env.constants.Unit.Barracks)
        for barracks in barracks_list:
            barracks.build(0)

    def get_default_action_wrapper(self):
        wrapper = None

        if self.game == GeneralizedBuildUnitsScenario.GAME_DEEP_RTS:
            wrapper = BuildUnitsDeepRTSActionWrapper() 
        elif self.game == GeneralizedBuildUnitsScenario.GAME_STARCRAFT_II:
            wrapper = BuildUnitsStarcraftIIActionWrapper()

        return wrapper 
