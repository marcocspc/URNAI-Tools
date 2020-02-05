import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from absl import app
from pysc2.env import sc2_env
from envs.sc2 import SC2Env
from agents.actions import sc2 as scaux
from statistics import mean

HOR_THRESHOLD = 2
VER_THRESHOLD = 2

PENDING_ACTIONS = []

def print_army_mean(obs):
    army = scaux.select_army(obs, sc2_env.Race.terran)
    xs = [unit.x for unit in army]
    ys = [unit.y for unit in army]

    print("Army position is [{x}, {y}]".format(x=int(mean[xs]),y=int(mean[ys])))

def move_left(obs):
    army = scaux.select_army(obs, sc2_env.Race.terran)
    xs = [unit.x for unit in army]
    ys = [unit.y for unit in army]

    new_army_x = int(mean[xs])
    new_army_y = int(mean[ys]) - HOR_THRESHOLD

    for unit in army:
        PENDING_ACTIONS.append(actions.RAW_FUNCTIONS.Move_pt("now", unit.tag, [new_army_x, new_army_y]))
        
def move_right(obs):
    army = scaux.select_army(obs, sc2_env.Race.terran)
    xs = [unit.x for unit in army]
    ys = [unit.y for unit in army]

    new_army_x = int(mean[xs])
    new_army_y = int(mean[ys]) + HOR_THRESHOLD

    for unit in army:
        PENDING_ACTIONS.append(actions.RAW_FUNCTIONS.Move_pt("now", unit.tag, [new_army_x, new_army_y]))

def move_down(obs):
    army = scaux.select_army(obs, sc2_env.Race.terran)
    xs = [unit.x for unit in army]
    ys = [unit.y for unit in army]

    new_army_x = int(mean[xs]) + VER_THRESHOLD
    new_army_y = int(mean[ys])

    for unit in army:
        PENDING_ACTIONS.append(actions.RAW_FUNCTIONS.Move_pt("now", unit.tag, [new_army_x, new_army_y]))

def move_up(obs):
    army = scaux.select_army(obs, sc2_env.Race.terran)
    xs = [unit.x for unit in army]
    ys = [unit.y for unit in army]

    new_army_x = int(mean[xs]) - VER_THRESHOLD
    new_army_y = int(mean[ys])

    for unit in army:
        PENDING_ACTIONS.append(actions.RAW_FUNCTIONS.Move_pt("now", unit.tag, [new_army_x, new_army_y]))


def main():
#    players = [sc2_env.Agent(sc2_env.Race.terran), sc2_env.Bot(sc2_env.Race.random, sc2_env.Difficulty.very_easy)]
    players = [sc2_env.Agent(sc2_env.Race.terran)]
    scii = SC2Env(map_name="CollectMineralShards", render=True, step_mul=16, players=players)
    episodes = 100
    steps = 99999999999999 

    for ep in range(episodes):
        print("Episode " + str(ep + 1))
        scii.reset()
        state = None

        for step in range(steps):
            print("Step " + str(step + 1))

            #ask for direction
            string = ''' Choose:
            1 - Up
            2 - Down
            3 - Left
            4 - Right

            '''

            #need to insert a way to choose actions
            action = int(input(string))

            if state == None:
                action = scaux._NO_OP
            else:
                #check if it is not at the limit of screen
                #move it to desired direction
                if action == 1:
                    action = move_up(state)
                elif action == 2:
                    action = move_down(state)
                elif action == 3:
                    action = move_left(state)
                elif action == 4:
                    action = move_right(state)

            if len(PENDING_ACTIONS) > 0:
                action = PENDING_ACTIONS.pop()

            state, reward, done = scii.step(scaux._NO_OP)

            print("Current state: ")
            print(state)

            #print its coordinates
            print_army_mean(obs)

            if done:
                break

import sys
from absl import flags
FLAGS = flags.FLAGS
FLAGS(sys.argv)
main()
