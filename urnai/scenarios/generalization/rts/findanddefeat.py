from urnai.scenarios.base.abscenario import ABScenario
from .collectables import GeneralizedCollectablesScenario
from urnai.utils.error import EnvironmentNotSupportedError
from urnai.agents.actions.scenarios.rts.generalization.findanddefeat import FindAndDefeatDeepRTSActionWrapper, FindAndDefeatStarcraftIIActionWrapper 
from agents.actions.base.abwrapper import ActionWrapper
from pysc2.lib import actions, features, units
from urnai.agents.actions import sc2 as scaux
from urnai.agents.rewards.default import PureReward
import numpy as np
from pysc2.env import sc2_env
from statistics import mean
import random
import DeepRTS as drts

class GeneralizedFindaAndDefeatScenario(GeneralizedCollectablesScenario):

    GAME_DEEP_RTS = "drts" 
    GAME_STARCRAFT_II = "sc2" 

    def __init__(self, game = GAME_DEEP_RTS, render=False, drts_map="total-64x64-playable-43x31-findanddefeat.json", sc2_map="FindAndDefeatZerglings", drts_number_of_players=2, drts_start_oil=99999, drts_start_gold=99999, drts_start_lumber=99999, drts_start_food=99999):
        super().__init__(game=game, render=render, drts_map=drts_map, sc2_map=sc2_map, drts_number_of_players=drts_number_of_players, drts_start_oil=drts_start_oil, drts_start_gold=drts_start_gold, drts_start_lumber=drts_start_lumber, drts_start_food=drts_start_food)

    def start(self):
        super().start()

    def step(self, action):
        if (self.game == GeneralizedCollectablesScenario.GAME_DEEP_RTS):
            if self.steps == 0:
                tiles = self.env.game.tilemap.tiles
                player = 1
                for i in range(len(tiles)):
                    if self.env.game.players[player].num_archer > 23: break
                    self.random_spawn_archer(player)

            state, reward, done = self.env.step(action)
            self.steps += 1
            return state, reward, done 

        elif (self.game == GeneralizedCollectablesScenario.GAME_STARCRAFT_II):
            self.steps += 1
            return self.env.step(action)

    def random_spawn_archer(self, player):
        tile = self.random_tile()

        while not tile.is_buildable:
            tile = self.random_tile()

        if self.env.game.players[player].num_archer < 23:
            self.env.game.players[player].spawn_unit(self.env.constants.Unit.Archer, tile)

    def get_default_action_wrapper(self):
        wrapper = None

        if self.game == GeneralizedCollectablesScenario.GAME_DEEP_RTS:
            wrapper = FindAndDefeatDeepRTSActionWrapper() 
        elif self.game == GeneralizedCollectablesScenario.GAME_STARCRAFT_II:
            wrapper = FindAndDefeatStarcraftIIActionWrapper()

        return wrapper 

