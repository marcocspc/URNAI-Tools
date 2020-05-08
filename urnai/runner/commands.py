import os, argparse, time 
import sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

from .base.runner import Runner
from shutil import copyfile
from tdd.reporter import Reporter as rp 

class DeepRTSRunner(Runner):

    COMMAND = 'drts' 
    OPT_COMMANDS = [
            {'command': '--drts-map', 'help': 'Map to install, uninstall or use on DeepRTS.', 'type' : str, 'metavar' : 'MAP_PATH', 'action' : 'store'},
            {'command': '--install', 'help': 'Install map on DeepRTS.', 'action' : 'store_true'},
            {'command': '--uninstall', 'help': 'Uninstall map on DeepRTS.', 'action' : 'store_true'},
            {'command': '--show-available-maps', 'help': 'Show installed maps on DeepRTS.', 'action' : 'store_true'},
            ]
    
    def __init__(self, parser, args):
        super().__init__(parser, args)

    def run(self):
        from envs.deep_rts import DeepRTSEnv
        import DeepRTS as drts

        drts_map_dir = os.path.dirname(os.path.realpath(drts.python.__file__)) + '/assets/maps' 

        if self.args.show_available_maps:
            self.show_available_maps(drts_map_dir);
        elif self.args.drts_map is not None:
            map_name = os.path.basename(self.args.drts_map)
            full_map_path = os.path.abspath(map_name)

            if self.args.install:
                self.install_map(full_map_path, drts_map_dir)
            elif self.args.uninstall:
                self.uninstall_map(full_map_path, drts_map_dir)
            else:
                self.install_map(full_map_path, drts_map_dir, force=True)

                rp.report("Starting DeepRTS using map " + map_name)
                drts = DeepRTSEnv(render=True,map=map_name)
                drts.reset()

                try:
                    while True:
                        drts.reset()
                        drts.step(15)
                        time.sleep(1)
                except KeyboardInterrupt:
                    rp.report("Bye!")
                        
        else:
            raise argparse.ArgumentError(message="--drts-map not informed.")
        

    def is_map_installed(self, drts_map_dir, map_name):
        return os.path.exists(drts_map_dir + os.sep + map_name)

    def install_map(self, map_path, drts_map_dir, force=False):
        if force or not self.is_map_installed(drts_map_dir, os.path.basename(map_path)):
            if not force:
                rp.report("{map} is not installed, installing on DeepRTS...".format(map=os.path.basename(map_path)))
            copyfile(map_path, drts_map_dir + os.sep + os.path.basename(map_path))
        else:
            rp.report("{map} is already installed.".format(map=os.path.basename(map_path)))


    def uninstall_map(self, map_path, drts_map_dir):
        if self.is_map_installed(drts_map_dir, os.path.basename(map_path)):
            os.remove(drts_map_dir + os.sep + os.path.basename(map_path))
            rp.report("{map} was removed.".format(map=os.path.basename(map_path)))
        else:
            rp.report("{map} is not installed.".format(map=os.path.basename(map_path)))
            
    def show_available_maps(self, drts_map_dir):
        rp.report('Available maps on DeepRTS:')
        rp.report(os.listdir(drts_map_dir))

class TrainerRunner(Runner):

    COMMAND = 'train'
    OPT_COMMANDS = [
            {'command': '--json-file', 'help': 'JSON solve file, with all the parameters to start the training.', 'type' : str, 'metavar' : 'JSON_FILE_PATH', 'action' : 'store'},
#TODO            {'command': '--build-training-file', 'help': 'Helper to build a solve json-file.', 'action' : 'store_true'},
            ]

    def __init__(self, parser, args):
        super().__init__(parser, args)

    def run(self):
        if self.args.json_file is not None:
            from urnai.trainers.jsontrainer import JSONTrainer

            trainer = JSONTrainer(self.args.json_file)
            trainer.start_training()
        #TODO
        #elif self.args.build_training_file:
        else:
            raise argparse.ArgumentError(message="You must specify at least a JSON file path to start training.")
