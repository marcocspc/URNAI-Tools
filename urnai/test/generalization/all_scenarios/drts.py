#from urnai.scenarios.generalization.rts.defeatenemies import GeneralizedDefeatEnemiesScenario as Scenario
#from urnai.scenarios.generalization.rts.findanddefeat import GeneralizedFindaAndDefeatScenario as Scenario
from urnai.scenarios.generalization.rts.collectables import GeneralizedCollectablesScenario as Scenario
import numpy as np
import sys,os

episodes = 100
steps = 1000
print_map = True

env = Scenario(game = Scenario.GAME_DEEP_RTS, render = True)
action_wrapper = env.get_default_action_wrapper()
np.set_printoptions(threshold=sys.maxsize)

text = "Choose:\n"
cont = 0
for action in action_wrapper.get_actions():
    text += "\t{} - {}\n".format(action, action_wrapper.get_action_name_str_by_int(action))

for ep in range(episodes):
    reward = 0
    done = False
    state = env.reset()
    print("Episode " + str(ep + 1))
    total_ep_reward = 0
    
    for step in range(steps):
            print("Step " + str(step + 1))

            print(env.env.render)
            
            action = None

            try:
                action = int(input(text))
            except ValueError:
                #No-action
                action = action_wrapper.noaction 

            if state is not None:
                action = action_wrapper.get_action(action, state)
            else:
                action = action_wrapper.noaction 

            state, reward, done = env.step(action)
            total_ep_reward += reward

            for player in env.env.game.players:
                idx = env.env.game.players.index(player)
                player_stats = '''Player {player} stats: 
Oil: {o}
Gold: {g}
Lumber: {l}
Food: {f}
Number of archers: {na}'''.format(player=idx+1,o=player.oil,g=player.gold,l=player.lumber,f=player.food,na=player.num_archer)
            if env.collectables_map is not None: 
                print("Map:\n{}".format(env.collectables_map))
                np.savetxt(os.path.expanduser("~") + "/temp.csv", env.collectables_map, fmt='%i',delimiter=",")
            print("Reward: {r}".format(r=reward))
            print("Total Episode Reward: {r}".format(r=total_ep_reward))
            print(player_stats)

            if done: break
