from urnai.runner.base.runner import Runner
from shutil import copyfile
from urnai.tdd.reporter import Reporter as rp 
from urnai.utils import drts_utils
from urnai.utils import sc2_utils
import os, sys
import argparse

class DeepRTSRunner(Runner):

    COMMAND = 'drts' 
    OPT_COMMANDS = [
            {'command': '--drts-map', 'help': 'Map to work with when using drts command.', 'type' : str, 'metavar' : 'MAP_PATH', 'action' : 'store'},
            {'command': '--extract-specs', 'help': 'This will export every map layer to a csv file. Other userful information will be on a JSON file. This switch works with both sc2 and drts commands.', 'action' : 'store_true'},
            {'command': '--drts-map-specs', 'help': 'Directory to work with when building a drts map.', 'type' : str, 'metavar' : 'DRTS_MAP_SPEC_PATH', 'action' : 'store'},
            {'command': '--build-map', 'help': 'This will build a map inside the directory informed with --drts-map-specs. If you need a template, you should use --extract-specs first. URNAI will generate the needed files from an existing DeepRTS map.', 'action' : 'store_true'},
            {'command': '--install', 'help': 'Install map on DeepRTS.', 'action' : 'store_true'},
            {'command': '--uninstall', 'help': 'Uninstall map on DeepRTS.', 'action' : 'store_true'},
            {'command': '--show-available-maps', 'help': 'Show installed maps on DeepRTS.', 'action' : 'store_true'},
            ]
    
    def __init__(self, parser, args):
        super().__init__(parser, args)

    def run(self):
        if self.args.show_available_maps:
            drts_utils.show_available_maps();
        elif self.args.drts_map is not None:
            map_name = os.path.basename(self.args.drts_map)
            full_map_path = os.path.abspath(map_name)

            if self.args.install:
                drts_utils.install_map(full_map_path)
            elif self.args.uninstall:
                drts_utils.uninstall_map(full_map_path)
            elif self.args.extract_specs:
                if len(os.listdir(".")) > 0:
                    answer = ""
                    while not (answer.lower() == "y" or answer.lower() == "n") :
                        answer = input("Current directory is not empty. Do you wish to continue? [y/n]")

                    if answer.lower() == "y":
                        rp.report("Extracting {map} features...".format(map=map_name))
                        drts_utils.extract_specs(map_name)
                else:
                    rp.report("Extracting {map} features...".format(map=map_name))
                    drts_utils.extract_specs(map_name)
            
            else:
                drts_utils.install_map(full_map_path, force=True)

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
        elif self.args.build_map:
            if self.args.drts_map_specs is not None:
                if len(os.listdir(self.args.drts_map_specs)) > 0:
                    drts_utils.build_map(self.args.drts_map_specs)
                else:
                    rp.report("DeepRTSMapSpecs directory is empty.")
            else:
                rp.report("--drts-map-specs weren't informed.")
        else:
            raise argparse.ArgumentError(message="--drts-map not informed.")


class TrainerRunner(Runner):

    COMMAND = 'train'
    OPT_COMMANDS = [
            {'command': '--json-file', 'help': 'JSON solve file, with all the parameters to start the training.', 'type' : str, 'metavar' : 'JSON_FILE_PATH', 'action' : 'store'},
#TODO            {'command': '--build-training-file', 'help': 'Helper to build a solve json-file.', 'action' : 'store_true'},
            {'command': '--play', 'help': 'Test agent, without training it, it will ignore train entry on json file.', 'action' : 'store_true'},
            ]

    def __init__(self, parser, args):
        super().__init__(parser, args)

    def run(self):
        if self.args.json_file is not None:
            from urnai.trainers.jsontrainer import JSONTrainer
            trainer = JSONTrainer(self.args.json_file)

            if self.args.play:
                trainer.start_training(play_only=True)
            else:
                trainer.start_training()
        #TODO
        #elif self.args.build_training_file:
        else:
            raise argparse.ArgumentError(message="You must specify at least a JSON file path to start training.")

class SC2Runner(Runner):

    COMMAND = 'sc2'
    OPT_COMMANDS = [
            {'command': '--sc2-map', 'help': 'Map to use on Starcraft II.', 'type' : str, 'metavar' : 'MAP_FILE_PATH', 'action' : 'store'},
            ]

    def run (self):
        #remove all args from sys.argv
        #this is needed to run sc2 env without errors
        for arg in self.args.__dict__.keys():
            arg = "--{}".format(arg.replace("_", "-"))
            if arg in sys.argv: sys.argv.remove(arg)


        if self.args.sc2_map is not None:
            if self.args.extract_specs:
                map_name = self.args.sc2_map
                if len(os.listdir(".")) > 0:
                    answer = ""
                    while not (answer.lower() == "y" or answer.lower() == "n") :
                        answer = input("Current directory is not empty. Do you wish to continue? [y/n]")

                    if answer.lower() == "y":
                        rp.report("Extracting {map} features...".format(map=map_name))
                        sc2_utils.extract_specs(map_name)
                else:
                    rp.report("Extracting {map} features...".format(map=map_name))
                    sc2_utils.extract_specs(map_name)
        else:
            rp.report("--sc2-map not informed.")
