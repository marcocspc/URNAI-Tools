import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from absl import app
from pysc2.env import sc2_env
from envs.sc2 import SC2Env
from agents.actions import sc2 as sca
from statistics import mean

HOR_THRESHOLD = 2
VER_THRESHOLD = 2


def move_left(obs, unit):
    army = sca.select_army(obs, sc2_env.Race.terran)
    xs = [unit.x for unit in army]
    ys = [unit.y for unit in army]

    new_army_x = int(mean[xs])
    new_army_y = int(mean[ys]) - HOR_THRESHOLD



def main():
    scii = SC2Env(map_name="CollectMineralShards", render=True, step_mul=16)
    episodes = 100
    steps = 100

    for ep in range(episodes):
        print("Episode " + str(ep + 1))
        scii.reset()

        for step in range(steps):
            print("Step " + str(step + 1))
            #need to insert a way to choose actions
            #select unit (or units)

            #get its coordinates

            #print its coordinates

            #ask for direction

            #check if it is not at the limit of screen
                #move it to desired direction

            state, reward, done = scii.step(action)

            print("Current state: ")
            print(state)

            if done:
                break

