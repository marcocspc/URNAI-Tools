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


class GeneralizedDefeatEnemiesScenario(GeneralizedFindaAndDefeatScenario):

    GAME_DEEP_RTS = "drts" 
    GAME_STARCRAFT_II = "sc2" 
    MAP_1 = "total-64x64-playable-21x13-defeatenemies_1.json"
    MAP_2 = "total-64x64-playable-21x13-defeatenemies_2.json"

    def __init__(self, game = GAME_DEEP_RTS, render=False, drts_map="total-64x64-playable-21x13-defeatenemies_1.json", sc2_map="DefeatRoaches", drts_start_oil=999999, drts_start_gold=999999, drts_start_lumber=999999, drts_start_food=999999):
        super().__init__(game=game, render=render, drts_map=drts_map, sc2_map=sc2_map, drts_number_of_players=2, drts_start_oil=drts_start_oil, drts_start_gold=drts_start_gold, drts_start_lumber=drts_start_lumber, drts_start_food=drts_start_food)
        self.map = GeneralizedDefeatEnemiesScenario.MAP_1

    def setup_map(self):
        if self.game == GeneralizedDefeatEnemiesScenario.GAME_DEEP_RTS:
            choice = random.randint(0, 1)

            if choice == 0:
                self.map = GeneralizedDefeatEnemiesScenario.MAP_1
            else:
                self.map = GeneralizedDefeatEnemiesScenario.MAP_2

            if self.env.map != self.map:
                self.env.change_map(self.map)

    def step(self, action):
        if (self.game == GeneralizedDefeatEnemiesScenario.GAME_DEEP_RTS):
            state, reward, done = self.env.step(action)
            return state, reward, done 

        elif (self.game == GeneralizedDefeatEnemiesScenario.GAME_STARCRAFT_II):
            return self.env.step(action)

    def get_default_action_wrapper(self):
        wrapper = None

        if self.game == GeneralizedDefeatEnemiesScenario.GAME_DEEP_RTS:
            wrapper = DefeatEnemiesDeepRTSActionWrapper() 
        elif self.game == GeneralizedDefeatEnemiesScenario.GAME_STARCRAFT_II:
            wrapper = DefeatEnemiesStarcraftIIActionWrapper()

        return wrapper 
