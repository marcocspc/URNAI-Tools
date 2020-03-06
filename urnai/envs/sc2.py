import numpy as np
from .base.abenv import Env
from pysc2.lib import actions
from pysc2.lib import features
from pysc2.lib import protocol
from pysc2.env.environment import StepType

# TODO: Add actions for specific races
# Keys that can be passed do SC2Env constructor to add specific actions.
# 'minigames': Adds actions used in all minigames
# 'minigames_all': Adds additional actions for minigames, which are not necessary to solve them
# 'all': Adds all actions, including outdated/unusable to current race/usable in certain situations
ACTIONS_MINIGAMES, ACTIONS_MINIGAMES_ALL, ACTIONS_ALL = 'minigames', 'minigames_all', 'all'

class SC2Env(Env):
    def __init__(
        self,
        map_name='Simple64',
        players=None,
        render=False,
        reset_done=True,
        spatial_dim=16,
        step_mul=8,
        game_steps_per_ep=0,
        obs_features=None,
        action_ids=ACTIONS_ALL    # action_ids is passed to the constructor as a key for the actions that the agent can use, but is converted to a list of IDs for these actions
    ):
        super().__init__(map_name, render, reset_done)

        self.step_mul = step_mul
        self.game_steps_per_ep = game_steps_per_ep
        self.spatial_dim = spatial_dim
        self.players = players

        self.start()

    
    def start(self):
        if self.env_instance is None:
            # Lazy loading pysc2 env
            from pysc2.env import sc2_env


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
        return self.parse_timestep(timestep)    #obs, reward, done


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
        
