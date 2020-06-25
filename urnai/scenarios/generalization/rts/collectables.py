import sys
from urnai.scenarios.base.abscenario import ABScenario
from urnai.utils.error import EnvironmentNotSupportedError
from urnai.agents.actions.base.abwrapper import ActionWrapper
from urnai.agents.actions.scenarios.rts.generalization.collectables import CollectablesDeepRTSActionWrapper, CollectablesStarcraftIIActionWrapper 
from pysc2.lib import actions, features, units
from agents.actions import sc2 as scaux
from agents.rewards.default import PureReward
import numpy as np
from pysc2.env import sc2_env
from urnai.envs.sc2 import SC2Env
from urnai.envs.deep_rts import DeepRTSEnv
from absl import flags
from statistics import mean
import random
import sys, os


class GeneralizedCollectablesScenario(ABScenario):

    GAME_DEEP_RTS = "drts" 
    GAME_STARCRAFT_II = "sc2" 

    def __init__(self, game = GAME_DEEP_RTS, render=False, drts_map="total-64x64-Playable-21x15-collectables.json", sc2_map="CollectMineralShards", drts_number_of_players=1, drts_start_oil=99999, drts_start_gold=99999, drts_start_lumber=99999, drts_start_food=99999):
        self.game = game
        if game == GeneralizedCollectablesScenario.GAME_DEEP_RTS:
            self.env = DeepRTSEnv(render=render, map=drts_map, updates_per_action = 12, number_of_players=drts_number_of_players, start_oil=drts_start_oil, start_gold=drts_start_gold, start_lumber=drts_start_lumber, start_food=drts_start_food)
        elif game == GeneralizedCollectablesScenario.GAME_STARCRAFT_II:
            FLAGS = flags.FLAGS
            FLAGS(sys.argv)
            players = [sc2_env.Agent(sc2_env.Race.terran)]
            self.env = SC2Env(map_name=sc2_map, render=render, step_mul=32, players=players)
        else:
            err = '''{} only supports the following environments:
    GeneralizedCollectablesScenario.GAME_DEEP_RTS
    GeneralizedCollectablesScenario.GAME_STARCRAFT_II'''.format(self.__class__.__name__)
            raise EnvironmentNotSupportedError(err)

        self.start()
        self.steps = 0

    def setup_map(self):
        if (self.game == GeneralizedCollectablesScenario.GAME_DEEP_RTS):
            self.collectables_map = self.set_collectable_map() 

    def get_player_units(self, player):
        units = []
        for unit in self.env.game.units:
            if unit.get_player().get_id() == player:
                units.append(unit)

        return units

    def start(self):
        self.env.start()
        self.done = self.env.done

    def step(self, action):
        if (self.game == GeneralizedCollectablesScenario.GAME_DEEP_RTS):
            if self.steps == 0:
                self.setup_map()

            state, reward, done = self.env.step(action)

            reward = 0

            player = 0
            tile_value = 0
            for unit in self.get_player_units(player):
                unit_x = unit.tile.x
                unit_y = unit.tile.y

                tile_value += self.collectables_map[unit_y, unit_x]
                if (tile_value > 0):
                    reward += tile_value 
                    self.collectables_map[unit_y, unit_x] = 0

            self.steps += 1
            return state, reward, done
        elif (self.game == GeneralizedCollectablesScenario.GAME_STARCRAFT_II):
            self.steps += 1
            return self.env.step(action)


    def close(self):
        self.env.close()
        self.done = self.env.done

    def reset(self):
        return self.env.reset()

    def restart(self):
        self.reset()

    def get_default_reward_builder(self):
        builder = PureReward() 
        #if self.game == GeneralizedCollectablesScenario.GAME_DEEP_RTS:
        #    builder = DeepRTSRewardBuilder()
        #elif self.game == GeneralizedCollectablesScenario.GAME_STARCRAFT_II:
        #    builder = StarcraftIIRewardBuilder()
        return builder

    def get_default_action_wrapper(self):
        wrapper = None

        if self.game == GeneralizedCollectablesScenario.GAME_DEEP_RTS:
            wrapper = CollectablesDeepRTSActionWrapper() 
        elif self.game == GeneralizedCollectablesScenario.GAME_STARCRAFT_II:
            wrapper = CollectablesStarcraftIIActionWrapper()

        return wrapper 

    def set_collectable_map(self):
        if self.game == GeneralizedCollectablesScenario.GAME_DEEP_RTS:
            width = self.env.game.map.map_width
            height = self.env.game.map.map_height
            map_ = np.zeros((width, height)) 

            for i in range(width):
                for j in range(height):
                    if self.env.game.tilemap.get_tile(j, i).is_walkable():
                        map_[i][j] = random.randint(0, 1)
                        if np.sum(map_) > 20:
                            break
                else:
                    continue
                break

        return map_
