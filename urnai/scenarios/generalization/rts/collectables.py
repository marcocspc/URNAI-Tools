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
from absl import flags
from statistics import mean


class GeneralizedCollectablesScenario(ABScenario):

    GAME_DEEP_RTS = "drts" 
    GAME_STARCRAFT_II = "sc2" 

    def __init__(self, game = GAME_DEEP_RTS, render=False, drts_map="10x8-collect_twenty.json", sc2_map="CollectMineralShards"):
        FLAGS = flags.FLAGS
        FLAGS(sys.argv)
        self.game = game
        if game == GeneralizedCollectablesScenario.GAME_DEEP_RTS:
            self.env = DeepRTSEnv(render=render, map=drts_map, updates_per_action = 12)
            self.collectables_map = self.set_collectable_map() 
        elif game == GeneralizedCollectablesScenario.GAME_STARCRAFT_II:
            players = [sc2_env.Agent(sc2_env.Race.terran)]
            self.env = SC2Env(map_name=sc2_map, render=render, step_mul=32, players=players)
        else:
            err = '''{} only supports the following environments:
    GeneralizedCollectablesScenario.GAME_DEEP_RTS
    GeneralizedCollectablesScenario.GAME_STARCRAFT_II'''.format(self.__class__.__name__)
            raise EnvironmentNotSupportedError(err)

    def start(self):
        self.env.start()
        self.done = self.env.done

    def step(self, action):
        if (self.game == GeneralizedCollectablesScenario.GAME_DEEP_RTS):
            state, done = self.env.step(action)

            unit_x = self.env.players[0].get_targeted_unit().tile.x
            unit_y = self.env.players[0].get_targeted_unit().tile.y

            reward = self.collectables_map[unit_y - 1, unit_x - 1]

            if (reward > 0):
                self.collectables_map[unit_y - 1, unit_x - 1] = 0

            return state, reward, done
        elif (self.game == GeneralizedCollectablesScenario.GAME_STARCRAFT_II):
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
        width, height = 10, 8
        map = np.zeros((height, width)) 

        for i in range(width):
            for j in range(height):
                map[i][j] = random.randint(0, 1)
                if np.sum(map) > 20:
                    break
            else:
                continue
            break

        return map
