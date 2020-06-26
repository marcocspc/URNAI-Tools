import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0,parentdir) 

from urnai.scenarios.base.abscenario import ABScenario
from .findanddefeat import GeneralizedFindaAndDefeatScenario 
from urnai.utils.error import EnvironmentNotSupportedError
from urnai.agents.actions.scenarios.rts.generalization.defeatenemies import DefeatEnemiesDeepRTSActionWrapper, DefeatEnemiesStarcraftIIActionWrapper 
from pysc2.lib import actions, features, units
from urnai.agents.actions import sc2 as scaux
from urnai.agents.rewards.default import PureReward
import numpy as np
from pysc2.env import sc2_env
from statistics import mean
import random
from urnai.envs.deep_rts import DeepRTSEnv


class GeneralizedDefeatEnemiesScenario(GeneralizedFindaAndDefeatScenario):

    GAME_DEEP_RTS = "drts" 
    GAME_STARCRAFT_II = "sc2" 
    MAP_1 = "map1"
    MAP_2 = "map2"

    MAP_1_PLAYER_X = 39
    MAP_2_PLAYER_X = 32
    Y_PLAYER_BASE = 33
    MAP_PLAYER_LOCATIONS = {
            MAP_1 : [
                {"x" : MAP_1_PLAYER_X, "y" : Y_PLAYER_BASE},
                {"x" : MAP_1_PLAYER_X, "y" : Y_PLAYER_BASE + 1},
                {"x" : MAP_1_PLAYER_X, "y" : Y_PLAYER_BASE + 2},
                {"x" : MAP_1_PLAYER_X, "y" : Y_PLAYER_BASE + 3},
                {"x" : MAP_1_PLAYER_X, "y" : Y_PLAYER_BASE + 4},
                {"x" : MAP_1_PLAYER_X, "y" : Y_PLAYER_BASE + 5},
                {"x" : MAP_1_PLAYER_X, "y" : Y_PLAYER_BASE + 6},
                {"x" : MAP_1_PLAYER_X, "y" : Y_PLAYER_BASE + 7},
                {"x" : MAP_1_PLAYER_X, "y" : Y_PLAYER_BASE + 8},
                ],
            MAP_2 : [
                {"x" : MAP_2_PLAYER_X, "y" : Y_PLAYER_BASE},
                {"x" : MAP_2_PLAYER_X, "y" : Y_PLAYER_BASE + 1},
                {"x" : MAP_2_PLAYER_X, "y" : Y_PLAYER_BASE + 2},
                {"x" : MAP_2_PLAYER_X, "y" : Y_PLAYER_BASE + 3},
                {"x" : MAP_2_PLAYER_X, "y" : Y_PLAYER_BASE + 4},
                {"x" : MAP_2_PLAYER_X, "y" : Y_PLAYER_BASE + 5},
                {"x" : MAP_2_PLAYER_X, "y" : Y_PLAYER_BASE + 6},
                {"x" : MAP_2_PLAYER_X, "y" : Y_PLAYER_BASE + 7},
                {"x" : MAP_2_PLAYER_X, "y" : Y_PLAYER_BASE + 8},
                ],
            }

    MAP_1_ENEMY_X = 30
    MAP_2_ENEMY_X = 41
    Y_ENEMY_BASE = 35
    MAP_ENEMY_LOCATIONS = {
            MAP_1 : [
                {"x" : MAP_1_ENEMY_X, "y" : Y_ENEMY_BASE},
                {"x" : MAP_1_ENEMY_X, "y" : Y_ENEMY_BASE + 1},
                {"x" : MAP_1_ENEMY_X, "y" : Y_ENEMY_BASE + 2},
                {"x" : MAP_1_ENEMY_X, "y" : Y_ENEMY_BASE + 3},
                ],
            MAP_2 : [
                {"x" : MAP_2_ENEMY_X, "y" : Y_ENEMY_BASE},
                {"x" : MAP_2_ENEMY_X, "y" : Y_ENEMY_BASE + 1},
                {"x" : MAP_2_ENEMY_X, "y" : Y_ENEMY_BASE + 2},
                {"x" : MAP_2_ENEMY_X, "y" : Y_ENEMY_BASE + 3},
                ],
            }

    def __init__(self, game = GAME_DEEP_RTS, render=False, drts_map="total-64x64-playable-21x13-defeatenemies.json", sc2_map="DefeatRoaches", drts_start_oil=999999, drts_start_gold=999999, drts_start_lumber=999999, drts_start_food=999999):
        super().__init__(game=game, render=render, drts_map=drts_map, sc2_map=sc2_map, drts_number_of_players=2, drts_start_oil=drts_start_oil, drts_start_gold=drts_start_gold, drts_start_lumber=drts_start_lumber, drts_start_food=drts_start_food)

    def step(self, action):
        if (self.game == GeneralizedDefeatEnemiesScenario.GAME_DEEP_RTS):
            if self.steps == 0:
                self.setup_map()
                self.spawn_army()

            state, reward, done = self.env.step(action)
            self.steps += 1
            return state, reward, done 

        elif (self.game == GeneralizedDefeatEnemiesScenario.GAME_STARCRAFT_II):
            self.steps += 1
            return self.env.step(action)

    def setup_map(self):
        if self.game == GeneralizedDefeatEnemiesScenario.GAME_DEEP_RTS:
            choice = random.randint(0, 1)
            #choice = 1 

            if choice == 0:
                self.map_spawn = GeneralizedDefeatEnemiesScenario.MAP_1
            else:
                self.map_spawn = GeneralizedDefeatEnemiesScenario.MAP_2

    def spawn_army(self):
        for coords in GeneralizedDefeatEnemiesScenario.MAP_PLAYER_LOCATIONS[self.map_spawn]:
            tile = self.env.game.tilemap.get_tile(coords['x'], coords['y'])
            self.env.game.players[0].spawn_unit(self.env.constants.Unit.Archer, tile)

        for coords in GeneralizedDefeatEnemiesScenario.MAP_ENEMY_LOCATIONS[self.map_spawn]:
            tile = self.env.game.tilemap.get_tile(coords['x'], coords['y'])
            self.env.game.players[1].spawn_unit(self.env.constants.Unit.Archer, tile)

    def get_default_action_wrapper(self):
        wrapper = None

        if self.game == GeneralizedDefeatEnemiesScenario.GAME_DEEP_RTS:
            wrapper = DefeatEnemiesDeepRTSActionWrapper() 
        elif self.game == GeneralizedDefeatEnemiesScenario.GAME_STARCRAFT_II:
            wrapper = DefeatEnemiesStarcraftIIActionWrapper()

        return wrapper 
