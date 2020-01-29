import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from absl import app
from pysc2.env import sc2_env
from envs.sc2 import SC2Env

scii = SC2Env(map_name="CollectMineralShards", render=True, step_mul=16)
episodes = 100
steps = 100

for ep in range(episodes):
    print("Episode " + str(ep + 1))
    scii.reset()

    for step in range(steps):
        print("Step " + str(step + 1))
        state, reward, done = scii.step()

        print("Current state: ")
        print(state)

        if done:
            break

