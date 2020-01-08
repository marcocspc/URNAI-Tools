import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0,parentdir)

from absl import app
from envs.ple import PLEEnv
from envs.trainer import Trainer
from agents.generic_agent import GenericAgent
from agents.actions.ple_wrapper import PLEWrapper
from agents.rewards.default import PureReward
from agents.states.ple import FlappyBirdState
from models.dql_tf import DQLTF
from ple.games.flappybird import FlappyBird

def main(unused_argv):
    trainer = Trainer()

    try:
        env = PLEEnv(_id="FlappyBird", game=FlappyBird(), render=True)

        action_wrapper = PLEWrapper(env)
        state_builder = FlappyBirdState()

        dq_network = DQLTF(action_wrapper, state_builder, 'urnai/models/saved/flappybird_dql')

        agent = GenericAgent(dq_network, PureReward())

        # Using Trainer to train and play with our agent.
        trainer.train(env, agent, max_steps=1000, save_steps=1000)
        trainer.play(env, agent, num_matches=100, max_steps=1000)
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    app.run(main)
