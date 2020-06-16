from urnai.envs.sc2 import SC2Env
from pysc2.env import sc2_env
from pysc2.lib import actions
import numpy as np
from absl import flags
import sys
import json

def get_sc2_env(map_name):
    FLAGS = flags.FLAGS
    FLAGS(sys.argv)
    players = [sc2_env.Agent(sc2_env.Race.terran)]
    env = SC2Env(map_name=map_name, render=False, step_mul=32, players=players)
    return env


def extract_specs(map_name):
    #start sc2 env
    env = get_sc2_env(map_name) 
    env.start()
    state, reward, done = env.step([actions.RAW_FUNCTIONS.no_op()])

    json_output = {}
    json_output["map_name"] = map_name
    json_output["map_shape"] = state.feature_minimap[0].shape

    with open(map_name +'_info.json', 'w') as outfile:
        outfile.write(json.dumps(json_output, indent=4))

    cont = 0
    for minimap in state.feature_minimap:
        map_csv = np.array(minimap).astype(int) 
        np.savetxt("feature_minimap_layer_{}.csv".format(cont), map_csv, fmt='%i',delimiter=",")
        cont += 1
