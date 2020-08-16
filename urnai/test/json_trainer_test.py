import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0,parentdir)

from urnai.trainers.jsontrainer import JSONTrainer

trainer = JSONTrainer("D:/UFRN/Star Craft II - Reinforcement Learning/URNAI-Tools/urnai/test/solves/solve_simple64_veryeasy.json")
trainer.start_training()
