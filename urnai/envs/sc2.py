import sys
import numpy as np
from utils.sc2_utils import get_sc2_race, get_sc2_difficulty
from absl import flags
from .base.abenv import Env
from pysc2.lib import actions
from pysc2.lib import features
from pysc2.lib import protocol
from pysc2.env.environment import StepType
from pysc2.env import sc2_env

class SC2Env(Env):
    def __init__(
        self,
        map_name='Simple64',
        players=None,
        player_race = 'terran',
        enemy_race = 'random',
        difficulty = 'very_easy',
        render=False,
        reset_done=True,
        spatial_dim=16,
        step_mul=8,
        game_steps_per_ep=0,
        obs_features=None,
    ):
        super().__init__(map_name, render, reset_done)

        FLAGS = flags.FLAGS
        FLAGS(sys.argv)

        self.step_mul = step_mul
        self.game_steps_per_ep = game_steps_per_ep
        self.spatial_dim = spatial_dim
        self.player_race = get_sc2_race(player_race)
        self.enemy_race = get_sc2_race(enemy_race)
        self.difficulty = get_sc2_difficulty(difficulty)
        self.players = [sc2_env.Agent(self.player_race), sc2_env.Bot(self.enemy_race, self.difficulty)]
        self.done = False

        self.start()

    
    def start(self):
        self.done = False
        if self.env_instance is None:

            self.env_instance = sc2_env.SC2Env(
                map_name=self.id,
                visualize=self.render,
                players=self.players,
                agent_interface_format=[
                    features.AgentInterfaceFormat(
                        action_space=actions.ActionSpace.RAW,
                        use_raw_units=True,
                        raw_resolution=64,
                        use_feature_units=True,
                        feature_dimensions=features.Dimensions(screen=64, minimap=64),
                    )
                ],
                step_mul=self.step_mul,
                game_steps_per_episode=self.game_steps_per_ep
            )

    
    def step(self, action):
        timestep = self.env_instance.step(action)
        obs, reward, done = self.parse_timestep(timestep)
        self.done = done
        return obs, reward, done


    def reset(self):
        timestep = self.env_instance.reset()
        obs, reward, done = self.parse_timestep(timestep)
        return obs

    
    def close(self):
        self.env_instance.close()

    
    def restart(self):
        self.close()
        self.reset()

    
    def parse_timestep(self, timestep):
        '''
        Returns a [Observation, Reward, Done] tuple parsed from a given timestep.
        '''
        ts = timestep[0]
        obs, reward, done = ts.observation, ts.reward, ts.step_type == StepType.LAST

        return obs, reward, done
        
