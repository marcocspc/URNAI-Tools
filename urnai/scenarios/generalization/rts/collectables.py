import sys, os
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


class GeneralizedCollectablesScenario(ABScenario):

    GAME_DEEP_RTS = "drts" 
    GAME_STARCRAFT_II = "sc2" 

    def __init__(self, game = GAME_DEEP_RTS, render=False, drts_map="total-64x64-playable-16x22-collectables.json", sc2_map="CollectMineralShards", drts_number_of_players=1, drts_start_oil=99999, drts_start_gold=99999, drts_start_lumber=99999, drts_start_food=99999):
        self.game = game
        self.steps = 0
        self.drts_hor_threshold = 3
        self.drts_ver_threshold = 3
        self.drts_action_previousunit = 0 
        self.drts_action_nextunit = 1
        self.drts_action_moveleft = 2
        self.drts_action_moveright = 3
        self.drts_action_moveup = 4
        self.drts_action_movedown = 5
        self.drts_action_moveupleft = 6
        self.drts_action_moveupright = 7
        self.drts_action_movedownleft = 8
        self.drts_action_movedownright = 9
        self.drts_action_attack = 10
        self.drts_action_harvest = 11
        self.drts_action_build0 = 12
        self.drts_action_build1 = 13
        self.drts_action_build2 = 14
        self.drts_action_noaction = 15

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

    def get_army_mean(self, player):
        xs = []
        ys = []

        for unit in self.get_player_units(player):
            try:
                if unit.id != 1 and unit.id != 2:
                    xs.append(unit.tile.x)
                    ys.append(unit.tile.y)
            except AttributeError as ae:
                if not "'NoneType' object has no attribute 'x'" in str(ae):
                    raise

        army_x = int(mean(xs))
        army_y = int(mean(ys))
        return army_x, army_y

    def solve_action(self, action):
        player = 0
        if action == self.drts_action_moveup:
            self.move_troops_up(player)
            return True
        elif action == self.drts_action_movedown:
            self.move_troops_down(player)
            return True
        elif action == self.drts_action_moveleft:
            self.move_troops_left(player)
            return True
        elif action == self.drts_action_moveright:
            self.move_troops_right(player)
            return True
        else:
            return False


    def move_troops_up(self, player):
        cur_x, cur_y = self.get_army_mean(player)

        new_x = cur_x
        new_y = cur_y - self.drts_ver_threshold

        self.env.game.players[player].right_click(new_x, new_y)

    def move_troops_down(self, player):
        cur_x, cur_y = self.get_army_mean(player)

        new_x = cur_x
        new_y = cur_y + self.drts_ver_threshold

        self.env.game.players[player].right_click(new_x, new_y)

    def move_troops_left(self, player):
        cur_x, cur_y = self.get_army_mean(player)

        new_x = cur_x - self.drts_hor_threshold
        new_y = cur_y

        self.env.game.players[player].right_click(new_x, new_y)

    def move_troops_right(self, player):
        cur_x, cur_y = self.get_army_mean(player)

        new_x = cur_x + self.drts_hor_threshold
        new_y = cur_y

        self.env.game.players[player].right_click(new_x, new_y)

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

            if self.solve_action(action):
                state, reward, done = self.env.step(self.drts_action_noaction)
            else:
                state, reward, done = self.env.step(action)

            player = 0

            reward = 0
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
        self.steps = 0
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

    def random_tile(self):
        tiles_len = len(self.env.game.tilemap.tiles)
        return self.env.game.tilemap.tiles[random.randint(0, tiles_len-1)]

    def set_collectable_map(self):
        if self.game == GeneralizedCollectablesScenario.GAME_DEEP_RTS:
            width = self.env.game.map.map_width
            height = self.env.game.map.map_height
            map_ = np.zeros((width, height)) 

            while np.sum(map_) <= 20:
                tile = self.random_tile()

                while not tile.is_walkable() or map_[tile.y, tile.x] != 0:
                    tile = self.random_tile()

                map_[tile.y, tile.x] = 1

            return map_
