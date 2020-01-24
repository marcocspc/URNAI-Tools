from .base.runner import Runner
from shutil import copyfile
import os
import argparse

class DeepRTSMapView(Runner):

    COMMANDS = [
            {'command': 'drts', 'help': 'Execute operations related to DeepRTS env.', 'type' : str},
            ]
    OPT_COMMANDS = [
            {'command': '--drts-map', 'help': 'Map to install, uninstall or use on DeepRTS.', 'type' : str},
            {'command': '--install', 'help': 'Install map on DeepRTS.', 'type' : bool},
            {'command': '--uninstall', 'help': 'Uninstall map on DeepRTS.', 'type' : bool},
            ]
    
    def __init__(self, parser, args):
        super().__init__(parser, args)

    def run(self):
        import os,sys,inspect
        currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        parentdir = os.path.dirname(currentdir)
        sys.path.insert(0,parentdir) 
        from envs.deep_rts import DeepRTSEnv
        import DeepRTS as drts

        if self.args.drts_map is not None:
            map_name = os.path.basename(self.args.drts_map)
            full_map_path = os.path.abspath('map_name')
            drts_map_dir = os.path.dirname(os.path.realpath(drts.python.__file__)) + '/assets/maps' 

            if self.args.install:
                self.install_map(full_map_path, drts_map_dir)
            elif self.args.uninstall:
                self.uninstall_map(full_map_path, drts_map_dir)
            else:
                if not self.is_map_installed(drts_map_dir, map_name):
                    self.install_map(full_map_path, drts_map_dir)

                print("Starting DeepRTS using map " + map_name)
                stamp = os.stat(full_map_path).st_mtime 
                drts = DeepRTSEnv(render=True,map=map_name)
                drts.reset()

                try:
                    while True:
                        current_stamp = os.stat(full_map_path).st_mtime 
                        if current_stamp != stamp:
                            stamp = current_stamp
                            drts.stop()
                            self.install_map(full_map_path, drts)
                            drts = DeepRTSEnv(render=True,map=map_name)
                            drts.reset()
                except KeyboardInterrupt:
                    print("Bye!")
        else:
            raise argparse.ArgumentError("--drts-map not informed.")
        

    def is_map_installed(self, drts_map_dir, map_name):
        return os.path.exists(drts_map_dir + os.sep + map_name)

    def install_map(self, map_path, drts_map_dir):
        print("{map} is not installed, installing on DeepRTS...".format(map=os.path.basename(map_path)))
        copyfile(map_name, drts_map_dir)

    def uninstall_map(self, map_path, drts_map_dir):
        if is_map_installed(drts_map_dir, map_path):
            os.remove(drts_map_dir + os.sep + map_path)
            print("{map} was removed.".format(map=os.path.basename(map_path)))
        else:
            print("{map} is not installed.".format(map=os.path.basename(map_path)))
            
