from .base.abenv import Env
from rlpytorch import load_module, argparse 
from os.path import expanduser
import sys

class MiniRTSEnv(Env):

    #TODO: add enemy AI to environment, check: MIGHT HAVE BEEN DONE, NEED TO BE TESTED 
    #https://github.com/facebookresearch/ELF/blob/master/train_minirts.sh and
    #https://github.com/facebookresearch/ELF/blob/master/eval_minirts.sh and
    #https://github.com/facebookresearch/ELF/blob/master/eval.py
    #TODO: TEST THIS!!!!!

    def __init__(self, minirts_path = expanduser("~") + "/ELF/rts/game_MC/game.py", enemyAI = True, ai_fskip = 50, model_fskip = 50):
        '''
            Initizalize attributes
        '''
        #remove .py from minirts path
        if minirts_path.endswith('.py'):
            minirts_path = minirts_path[:-3]

        game = load_module(minirts_path).Loader()

        #This might add an AI to the game
        argparse.ArgumentParser()
        cmd = '--num_games 1024 --batchsize 128 --tqdm --load ${MODEL} --gpu 0 --players "fs=' + str(model_fskip) + ',type=AI_NN;fs=+' str(ai_fskip) ' +,type=AI_SIMPLE" --eval_stats winrate --num_eval $NUM_EVAL --additional_labels id,last_terminal,seq --shuffle_player --num_frames_in_state 1 --greedy --keys_in_reply V'
        game.args.Load(argparse.ArgumentParser(), game, cmd_line=cmd)

        self.rts = game.initialize()

        self.observation = None
        self.actor_action = None
        self.done = True
        self.chosen_action = self._act_callback

    def start(self):
        '''
            Register action callback
            Start minirts
        '''

        self.rts.reg_callback("actor", self.chosen_action)
        self.rts.Start()
        self.done = False

    def step(self, action):
        '''
            Executes action and return observation, done
        '''

        self.actor_action = action
        self.rts.Run()

        return self.observation,self.done

    def close(self):
        '''
            Stops minirts
        '''
        self.rts.Stop()
        self.done = True

    def reset(self):
        '''
            Restart environment and returns initial observation
            TODO: check if self.start() modifies self.observation value
        '''

        self.close()
        self.observation = None
        self.actor_action = None
        self.start()

        return self.observation

    def restart(self):
        self.reset()


    def _act_callback(self, batch):
        self.observation = batch
        return self.actor_action







