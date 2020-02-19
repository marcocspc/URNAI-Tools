import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0,parentdir)

from envs.deep_rts import DeepRTSEnv
import random

episodes = 100
steps = 1000 
drts = DeepRTSEnv(render=True, map='10x8-collect_twenty.json', updates_per_action = 12)

for ep in range(episodes):
    print("Episode " + str(ep + 1))
    drts.reset()

    for step in range(steps):
        print("Step " + str(step + 1))
        
        text = '''
            Choose:
                PreviousUnit = 1,
                NextUnit = 2,
                MoveLeft = 3,
                MoveRight = 4,
                MoveUp = 5,
                MoveDown = 6,
                MoveUpLeft = 7,
                MoveUpRight = 8,
                MoveDownLeft = 9,
                MoveDownRight = 10,
                Attack = 11,
                Harvest = 12,
                Build0 = 13,
                Build1 = 14,
                Build2 = 15,
                NoAction = 16
        '''

        action = None

        try:
            action = int(input(text)) - 1
        except ValueError:
            action = 15

        state, done = drts.step(action)

        print("Current state: ")
        print(state)
        print("Player 1 selected unit:")
        print(drts.players[0].get_targeted_unit())
        print("Unit coordinates: {x}, {y}".format(x=drts.players[0].get_targeted_unit().tile.x,y=drts.players[0].get_targeted_unit().tile.y))
        print("Some Player 1 stats:")
        print("Oil: {oil}".format(oil=drts.players[0].oil))
        print("Gold: {gold}".format(gold=drts.players[0].gold))
        print("Food: {food}".format(food=drts.players[0].food))
        print("Lumber: {lumber}".format(lumber=drts.players[0].lumber))


        print("FPS: " + str(drts.game.get_fps()))

        if done:
            break
