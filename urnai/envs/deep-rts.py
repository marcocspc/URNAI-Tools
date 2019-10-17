from pyDeepRTS import Game
from .base.abenv import Env
import os

#TODO: add a way to view game main window if render = True
#TODO: add enemy AI

class DeepRTSEnv(Env):
    
    def __init__(self), map = '10x10-2v2.json', render = False):

        self.map = map
        self.render = render

        if (os.path.isdir('./assets')):
            #Setup game, informing map
            self.game = Game(self.map)

            #Add two players
            self.player1 = self.game.add_player()
            self.player2 = self.game.add_player()

            #Set FPS and UPS limits
            self.game.set_max_fps(10000000)
            self.game.set_max_ups(10000000)

        else:
            raise Exception("Directory 'assets' not found. Please obtain it using:\n\n" +
                    "git clone https://github.com/cair/deep-rts.git")

    def start(self):
        #Set done
        self.done = False

        #Start DeepRTS
        self.game.start()

    def step(self, action):
        #Update game clock
        self.game.tick()

        #Process game state
        self.game.update()

        #If render is enabled, draw state to graphic
        if (self.render):
            self.game.render()

        #Get current state
        state = self.game.state

        #Make player1 act
        self.player1.do_action(action)

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
