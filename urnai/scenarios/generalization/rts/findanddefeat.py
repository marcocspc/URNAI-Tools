import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0,parentdir) 

from scenario.base.abscenario import ABScenario
from .collectables import GeneralizedCollectablesScenario
from util.error import EnvironmentNotSupportedError
from agents.actions.base.abwrapper import ActionWrapper
from pysc2.lib import actions, features, units
from urnai.agents.actions import sc2 as scaux
from urnai.agents.rewards.default import PureReward
import numpy as np


class GeneralizedFindaAndDefeatScenario(GeneralizedCollectablesScenario):

    GAME_DEEP_RTS = "drts" 
    GAME_STARCRAFT_II = "sc2" 
    SCII_HOR_THRESHOLD = 2
    SCII_VER_THRESHOLD = 2
    MAXIMUM_ATTACK_RANGE = 3

    def __init__(self, game = GeneralizedFindaAndDefeatScenario.GAME_DEEP_RTS, render=False):
        self.game = game
        if game == GeneralizedCollectablesScenario.GAME_DEEP_RTS:
            self.env = DeepRTSEnv(render=render, map='26x14-find_and_defeat.json', updates_per_action = 12)
        elif game == GeneralizedCollectablesScenario.GAME_STARCRAFT_II:
            self.env = SC2Env(map_name="FindAndDefeatZerglings", render=render, step_mul=32, players=players)
        else:
            err = '''{} only supports the following environments:
    GeneralizedCollectablesScenario.GAME_DEEP_RTS
    GeneralizedCollectablesScenario.GAME_STARCRAFT_II'''.format(self.__class__.__name__)
            raise EnvironmentNotSupportedError(err)

    def start(self):
        if (self.game == GeneralizedCollectablesScenario.GAME_DEEP_RTS):
            for i in range(25):
                self.random_spawn_unit(7, self.env.game, 1)
        super().start()


    def step(self, action):
        if (self.game == GeneralizedCollectablesScenario.GAME_DEEP_RTS):
            state, reward, done = self.env.step(action)

        elif (self.game == GeneralizedCollectablesScenario.GAME_STARCRAFT_II):
            return self.env.step(action)

    def random_spawn_unit(self, drts_unit, drts_game, player):
        tile_map_len = len(drts_game.tilemap.tiles)
        tile = drts_game.tilemap.tiles[random.randint(0, tile_map_len - 1)]

        while not tile.is_buildable():
            tile = drts_game.tilemap.tiles[random.randint(0, tile_map_len - 1)]

        drts_game.players[player].spawn_unit(drts.constants.Unit.Archer, tile)


class DeepRTSActionWrapper(ActionWrapper):
    def __init__(self):
        self.move_number = 0

        moveleft = 2
        moveright = 3
        moveup = 4
        movedown = 5
        attack = 10

        self.actions = [moveleft, moveright, moveup, movedown, attack] 

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
        self.attack = 4

        self.actions = [self.moveleft, self.moveright, self.moveup, self.movedown, self.attack] 
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
        elif action_idx == self.attack:
            self.attack_(obs)
    
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

    def get_nearest_enemy_unit_inside_radius(self, x, y, obs, radius):
        enemy_army = [unit for unit in obs.raw_units if unit.owner != 1] 

        closest_dist = 9999999999999 
        closest_unit = None
        for unit in enemy_army:
            xaux = unit.x
            yaux = unit.y

            dist = abs(math.hypot(x - xaux, y - yaux))
            print("dist "+str(dist))

            if dist <= closest_dist and dist <= radius:
                closest_dist = dist
                closest_unit = unit

        if closest_unit is not None:
            return closest_unit

    def attack_nearest_inside_radius(self, obs, radius):
        #get army coordinates
        army = scaux.select_army(obs, sc2_env.Race.terran)
        xs = [unit.x for unit in army]
        ys = [unit.y for unit in army]
        army_x = int(mean(xs))
        army_y = int(mean(ys)) - VER_THRESHOLD

        #get nearest unit
        enemy_unit = self.get_nearest_enemy_unit_inside_radius(army_x, army_y, obs, radius)

        #tell each unit in army to attack nearest enemy
        for unit in army:
            self.pending_actions.append(actions.RAW_FUNCTIONS.Attack_pt("now", unit.tag, [enemy_unit.x, enemy_unit.y]))

    def attack_(self, obs):
        self.attack_nearest_inside_radius(obs, GeneralizedFindaAndDefeatScenario.MAXIMUM_ATTACK_RANGE)
