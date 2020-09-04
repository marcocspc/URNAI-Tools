import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0,parentdir)

from urnai.trainers.jsontrainer import JSONTrainer

for filename in os.listdir(currentdir):
    if filename.endswith("json"):
        trainer = JSONTrainer(currentdir + os.path.sep + filename)
        trainer.start_training()
