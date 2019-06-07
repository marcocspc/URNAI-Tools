from vizdoom import * 
from .base.abenv import Env

class VizdoomEnv(Env):

    RES_640X480 = ScreenResolution.RES_640X480 
    RES_160X120 = ScreenResolution.RES_160X120 

    def __init__(self, wad, doommap="map01", _id="vizdoom", render=True, reset_done=True, num_steps=1000, res=RES_160X120):
        if wad != None and wad != "":
            super().__init__(_id, render, reset_done)
            self.num_steps = num_steps
            self.wad = wad
            self.doommap = doommap
            self.game = DoomGame()
            self.res = res
        else:
            raise Exception("A wad file is needed.")

    def start(self):
        '''
            Start vizdoom 
        '''

        #Init object, set wad and map
        self.game.set_doom_scenario_path(self.wad)
        if (self.doommap != None):
            self.game.set_doom_map(self.doommap)

        #Set available controls
        self.game.add_available_button(Button.ATTACK)
        self.game.add_available_button(Button.USE)
        self.game.add_available_button(Button.MOVE_FORWARD)
        self.game.add_available_button(Button.MOVE_BACKWARD)
        self.game.add_available_button(Button.MOVE_RIGHT)
        self.game.add_available_button(Button.MOVE_LEFT)
        self.game.add_available_button(Button.TURN_RIGHT)
        self.game.add_available_button(Button.TURN_LEFT)

        #Set which variables will be set in observation
        #self.game.get_state().game_variables will be a nparray
        #where every position will be one of the variables
        #below. All variables are floats.
        #The list is ordered on the same order the variable
        #are added here
        self.game.add_available_game_variable(GameVariable.KILLCOUNT)
        self.game.add_available_game_variable(GameVariable.ITEMCOUNT)
        self.game.add_available_game_variable(GameVariable.SECRETCOUNT)
        self.game.add_available_game_variable(GameVariable.DEATHCOUNT)
        self.game.add_available_game_variable(GameVariable.HITCOUNT)
        self.game.add_available_game_variable(GameVariable.HITS_TAKEN)
        self.game.add_available_game_variable(GameVariable.DAMAGECOUNT)
        self.game.add_available_game_variable(GameVariable.DAMAGE_TAKEN)
        self.game.add_available_game_variable(GameVariable.HEALTH)
        self.game.add_available_game_variable(GameVariable.ARMOR)
        self.game.add_available_game_variable(GameVariable.DEAD)
        self.game.add_available_game_variable(GameVariable.ATTACK_READY)
        self.game.add_available_game_variable(GameVariable.SELECTED_WEAPON)
        self.game.add_available_game_variable(GameVariable.SELECTED_WEAPON_AMMO)
        self.game.add_available_game_variable(GameVariable.POSITION_X)
        self.game.add_available_game_variable(GameVariable.POSITION_Y)
        self.game.add_available_game_variable(GameVariable.POSITION_Z)

        #Some graphic properties
        self.game.set_screen_resolution(self.res)
        self.game.set_screen_format(ScreenFormat.RGB24)
        self.game.set_render_hud(True)
        self.game.set_render_crosshair(False)
        self.game.set_render_weapon(True)
        self.game.set_render_decals(False)
        self.game.set_render_particles(False)
        self.game.set_window_visible(self.render)

        #Set reward -1 for every movement the agent makes
        self.game.set_living_reward(-1)


        #Set episode limit and a little delay to spawn monsters
        self.game.set_episode_timeout(self.num_steps)
        self.game.set_episode_start_time(0)

        self.game.init()

    def step(self, action):
        '''
            Executes action and return observation, reward and done
        '''

        reward = self.game.make_action(action)
        done = self.game.is_episode_finished()
        state = self.game.get_state()
        if state != None:
            self.observation = state
        
        
        return self.observation, reward, done

    def reset(self):
        '''
            Resets the game to a new episode
        '''
        self.close()
        self.start()
        if self.game.get_state() != None:
            self.observation = self.game.get_state()
        return self.observation

    def close(self):
        ''' 
            Stops vizdoom
        '''
        if self.game != None:
            self.game.close()

    def restart(self):
        self.reset()

