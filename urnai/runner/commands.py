from .base.runner import Runner
from shutil import copyfile
import os

class DeepRTSMapView(Runner):

    COMMAND = 'drtsmapview'
    
    def __init__(self, parser, args):
        super().__init__(parser, args)

    def run(self):
        import os,sys,inspect
        currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        parentdir = os.path.dirname(currentdir)
        sys.path.insert(0,parentdir) 
        from envs.deep_rts import DeepRTSEnv
        import DeepRTS as drts

        if not self.is_map_installed(self.args.map, drts):
            self.install_map(self.args.map, drts)

        map_name = os.path.basename(self.args.map)

        if (self.args.map is not None): 
            print("Starting DeepRTS using map " + map_name)
            stamp = os.stat(self.args.map).st_mtime 
            drts = DeepRTSEnv(render=True,map=map_name)
            drts.reset()

            try:
                while True:
                    current_stamp = os.stat(self.args.map).st_mtime 
                    if current_stamp != stamp:
                        stamp = current_stamp
                        drts.stop()
                        self.install_map(self.args.map, drts)
                        drts = DeepRTSEnv(render=True,map=map_name)
                        drts.reset()
            except KeyboardInterrupt:
                print("Bye!")
        else:
            self.parser.error("--map was not informed.")
        

    def is_map_installed(self, map, drts):
        map_name = os.path.basename(map)
        maps_folder = os.path.dirname(os.path.realpath(drts.python.__file__)) + '/assets/maps' 
        should_exist = maps_folder + os.path.sep + map_name

        return os.path.exists(should_exist)

    def install_map(self, map, drts):
        map_name = os.path.basename(map)
        maps_folder = os.path.dirname(os.path.realpath(drts.python.__file__)) + '/assets/maps' 
        copy_to = maps_folder + os.path.sep + map_name

        print("{map} is not installed, installing on DeepRTS...".format(map=map_name))
        copyfile(map, copy_to)
