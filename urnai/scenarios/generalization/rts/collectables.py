import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0,parentdir) 

from scenario.base.abscenario import ABScenario
from util.error import EnvironmentNotSupportedError
from agents.actions.base.abwrapper import ActionWrapper
from pysc2.lib import actions, features, units
from agents.actions import sc2 as scaux
from agents.rewards.default import PureReward
import numpy as np


class GeneralizedCollectablesScenario(ABScenario):

    GAME_DEEP_RTS = 0
    GAME_STARCRAFT_II = 0
    SCII_HOR_THRESHOLD = 2
    SCII_VER_THRESHOLD = 2

    def __init__(self, game = GeneralizedCollectablesScenario.GAME_DEEP_RTS, render=False):
        self.game = game
        if game == GeneralizedCollectablesScenario.GAME_DEEP_RTS:
            self.env = DeepRTSEnv(render=render, map='10x8-collect_twenty.json', updates_per_action = 12)
            self.collectables_map = self.set_collectable_map() 
        elif game == GeneralizedCollectablesScenario.GAME_STARCRAFT_II:
            self.env = SC2Env(map_name="CollectMineralShards", render=render, step_mul=32, players=players)
        else:
            err = '''Collectables only supports the following environments:
    GeneralizedCollectablesScenario.GAME_DEEP_RTS
    GeneralizedCollectablesScenario.GAME_STARCRAFT_II'''
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
        self.close()
        self.start()

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
            wrapper = DeepRTSActionWrapper()
        elif self.game == GeneralizedCollectablesScenario.GAME_STARCRAFT_II:
            wrapper = StarcraftIIActionWrapper()

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


class DeepRTSActionWrapper(ActionWrapper):
    def __init__(self):
        self.move_number = 0

        moveleft = 2
        moveright = 3
        moveup = 4
        movedown = 5

        self.actions = [moveleft, moveright, moveup, movedown] 

    def is_action_done(self):
        return True

    def reset(self):
        self.move_number = 0

    def get_actions(self):
        return self.actions
    
    def get_excluded_actions(self, obs):        
        return []

    def get_action(self, action_idx, obs):
        return self.actions[action_idx]

class StarcraftIIActionWrapper(ActionWrapper):
    def __init__(self):
        self.move_number = 0

        self.moveleft = 0
        self.moveright = 1
        self.moveup = 2
        self.movedown = 3

        self.actions = [self.moveleft, self.moveright, self.moveup, self.movedown] 
        self.pending_actions = []

    def is_action_done(self):
        return True

    def reset(self):
        self.move_number = 0

    def get_actions(self):
        return self.actions
    
    def get_excluded_actions(self, obs):        
        return []

    def get_action(self, action_idx, obs):
        if len(self.pending_actions) > 0:
            return [self.pending_actions.pop()]
        else:
            self.solve_action(action_idx, obs)
            return [actions.RAW_FUNCTIONS.no_op()]

    def solve_action(self, action_idx, obs):
        if action_idx == self.moveleft:
            self.move_left(obs)
        elif action_idx == self.moveright:
            self.move_right(obs)
        elif action_idx == self.moveup:
            self.move_up(obs)
        elif action_idx == self.movedown:
            self.move_down(obs)
    
    def move_left(self, obs):
        army = scaux.select_army(obs, sc2_env.Race.terran)
        xs = [unit.x for unit in army]
        ys = [unit.y for unit in army]

        new_army_x = int(mean(xs)) - GeneralizedCollectablesScenario.SCII_HOR_THRESHOLD
        new_army_y = int(mean(ys))

        for unit in army:
            self.pending_actions.append(actions.RAW_FUNCTIONS.Move_pt("now", unit.tag, [new_army_x, new_army_y]))
            
    def move_right(self, obs):
        army = scaux.select_army(obs, sc2_env.Race.terran)
        xs = [unit.x for unit in army]
        ys = [unit.y for unit in army]

        new_army_x = int(mean(xs)) + GeneralizedCollectablesScenario.SCII_HOR_THRESHOLD
        new_army_y = int(mean(ys))

        for unit in army:
            self.pending_actions.append(actions.RAW_FUNCTIONS.Move_pt("now", unit.tag, [new_army_x, new_army_y]))

    def move_down(self, obs):
        army = scaux.select_army(obs, sc2_env.Race.terran)
        xs = [unit.x for unit in army]
        ys = [unit.y for unit in army]

        new_army_x = int(mean(xs))
        new_army_y = int(mean(ys)) + GeneralizedCollectablesScenario.SCII_VER_THRESHOLD

        for unit in army:
            self.pending_actions.append(actions.RAW_FUNCTIONS.Move_pt("now", unit.tag, [new_army_x, new_army_y]))

    def move_up(self, obs):
        army = scaux.select_army(obs, sc2_env.Race.terran)
        xs = [unit.x for unit in army]
        ys = [unit.y for unit in army]

        new_army_x = int(mean(xs))
        new_army_y = int(mean(ys)) - GeneralizedCollectablesScenario.SCII_VER_THRESHOLD

        for unit in army:
            self.pending_actions.append(RAW_FUNCTIONS.Move_pt("now", unit.tag, [new_army_x, new_army_y]))

