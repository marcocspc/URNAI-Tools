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


class GeneralizedDefeatEnemiesScenario(GeneralizedFindaAndDefeatScenario):

    GAME_DEEP_RTS = "drts" 
    GAME_STARCRAFT_II = "sc2" 

    def __init__(self, game = GAME_DEEP_RTS, render=False, drts_map="26x14-find_and_defeat.json", sc2_map="DefeatRoaches"):
        super().__init__(game=game, render=render, drts_map=drts_map, sc2_map=sc2_map)


    def step(self, action):
        if (self.game == GeneralizedDefeatEnemiesScenario.GAME_DEEP_RTS):
            state, reward, done = self.env.step(action)
            return state, reward, done 

        elif (self.game == GeneralizedDefeatEnemiesScenario.GAME_STARCRAFT_II):
            return self.env.step(action)

    def random_spawn_unit(self, drts_unit, drts_game, player):
        tile_map_len = len(drts_game.tilemap.tiles)
        tile = drts_game.tilemap.tiles[random.randint(0, tile_map_len - 1)]

        while not tile.is_buildable():
            tile = drts_game.tilemap.tiles[random.randint(0, tile_map_len - 1)]

        drts_game.players[player].spawn_unit(drts.constants.Unit.Archer, tile)

    def get_default_action_wrapper(self):
        wrapper = None

        if self.game == GeneralizedDefeatEnemiesScenario.GAME_DEEP_RTS:
            wrapper = DefeatEnemiesDeepRTSActionWrapper() 
        elif self.game == GeneralizedDefeatEnemiesScenario.GAME_STARCRAFT_II:
            wrapper = DefeatEnemiesStarcraftIIActionWrapper()

        return wrapper 

