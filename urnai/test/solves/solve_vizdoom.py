import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0,parentdir)

from absl import app
from envs.vizdoom import VizdoomEnv
from envs.trainer import Trainer
from agents.generic_agent import GenericAgent
from agents.actions.vizdoom_wrapper import VizdoomHealthGatheringWrapper
from agents.rewards.vizdoom import VizDoomHealthGatheringReward  
from agents.states.vizdoom import TFVizDoomHealthGatheringState
from models.dql_tf import DQLTF

def main(unused_argv):
    trainer = Trainer()

    try:
        env = VizdoomEnv("utils/vizdoomwads/health_gathering.wad", render=True, doommap=None, res=VizdoomEnv.RES_640X480)

        action_wrapper = VizdoomHealthGatheringWrapper(env)
        state_builder = TFVizDoomHealthGatheringState()

        model = DQLTF(action_wrapper, state_builder, '/Applications/')

        agent = GenericAgent(model, VizDoomHealthGatheringReward())

        # Using Trainer to train and play with our agent.
        trainer.train(env, agent, num_episodes=10000, max_steps=2100, save_steps=100)
        trainer.play(env, agent, num_matches=50, max_steps=2100)
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    app.run(main)
