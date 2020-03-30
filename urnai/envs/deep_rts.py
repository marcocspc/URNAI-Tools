from .base.abenv import Env
import os
import DeepRTS as drts
from DeepRTS import Engine
from DeepRTS.Engine import Constants, UnitManager 
from DeepRTS.python import Config, Game 

import sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

from utils.error import MapNotFoundError  

#TODO: add enemy AI

class DeepRTSEnv(Env):

    '''
       DeepRTS.python.Config Defaults: https://github.com/cair/deep-rts/blob/master/DeepRTS/python/_py_config.py
       DeepRTS.Engine.Config.defaults() : https://github.com/cair/deep-rts/blob/master/src/Config.h
       Possible actions: https://github.com/cair/deep-rts/blob/master/src/Constants.h
       Player class: https://github.com/cair/deep-rts/blob/master/bindings/Player.cpp
       Unit class: https://github.com/cair/deep-rts/blob/master/bindings/Unit.cpp
       Engine.Config.defaults(): https://github.com/cair/deep-rts/blob/master/src/Config.h
       Engine.Config options: https://github.com/cair/deep-rts/blob/master/bindings/Config.cpp
    '''

    MAP_MICRO = Config.Map.TEN
    MAP_SMALL = Config.Map.FIFTEEN
    MAP_MEDIUM = Config.Map.TWENTYONE
    MAP_BIG = Config.Map.THIRTYONE
    MAP_BIG_MORE_PLAYERS = Config.Map.THIRTYONE_FOUR
    MAO_BIG_MAX_PLAYERS = Config.Map.THIRTYONE_SIX
    
    def __init__(self, map = Config.Map.TEN, render = False, 
            max_fps = 1000000, max_ups = 1000000, play_audio = False, 
            number_of_players = 1, updates_per_action = 1, flatten_state = True,
            drts_engine_config = None,
            start_oil=0, start_gold=1500, start_lumber=750, start_food = 1):

        if self.is_map_installed(map):
            self.map = map
        else:
            err = "Map {map_name} was not found on DeepRTS maps folder.".format(map_name=map)
            raise MapNotFoundError(map)

        self.render = render
        self.play_audio = play_audio
        self.updates_per_action = updates_per_action
        self.flatten_state = flatten_state
        self.number_of_players = number_of_players
        self.max_fps = max_fps
        self.max_ups = max_ups
        self.unit_manager = UnitManager
        self.constants = Constants 

        self.gui_config = Config(
            render=True,
            view=self.render,
            inputs=False,
            caption=False,
            unit_health=True,
            unit_outline=True,
            unit_animation=True,
            audio=self.play_audio,
            audio_volume=50
        )

        if drts_engine_config == None:
            self.engine_config = Engine.Config.defaults()
            self.engine_config.set_start_oil(start_oil)
            self.engine_config.set_start_gold(start_gold)
            self.engine_config.set_start_lumber(start_lumber)
            self.engine_config.set_start_food(start_food)
            self.engine_config.set_archer(True)
            self.engine_config.set_instant_town_hall(False)
            self.engine_config.set_barracks(True)
        else:
            self.engine_config = drts_engine_config

        self.game = Game(
            self.map,
            n_players = self.number_of_players,
            engine_config = self.engine_config,
            gui_config = self.gui_config,
            terminal_signal = False
        )
        self.game.set_max_fps(self.max_fps)
        self.game.set_max_ups(self.max_ups)
 
        self.players = self.game.players

    def start(self):
        #Set done
        self.done = False

        #Start DeepRTS
        self.game.start()
        self.game.reset()


    def step(self, action):
        #select first player
        #you can do self.game.selected_player
        player = self.game.players[0]

        #make player do that action
        #actions are values between 1 and 16
        #but in general models give values between
        #0 and 15, so that's why action value
        #is action + 1
        player.do_action(action + 1)

        #update game state
        for i in range(self.updates_per_action):
            self.game.update()

        #get game state, flattened or not
        state = None
        if self.flatten_state:
            state = self.game.get_state().flatten()
        else:
            state = self.game.get_state()

        #make game update its internal graphics
        #and show windows, if it was configured to
        self.game.render()
        if self.render:
            self.game.view()

        #Return observation and done
        return state, self.done

    def close(self):
        #Stop DeepRTS
        self.game.stop()

        #Set done
        self.done = True

    def reset(self):
        self.close()
        self.start()

    def restart(self):
        self.reset()

    def is_map_installed(self, map_name):
        drts_map_dir = os.path.dirname(os.path.realpath(drts.python.__file__)) + '/assets/maps' 
        return os.path.exists(drts_map_dir + os.sep + map_name)
