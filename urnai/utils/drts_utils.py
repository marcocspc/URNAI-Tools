from envs.deep_rts import DeepRTSEnv
import DeepRTS as drts

DRTS_MAPS_DIR = os.path.dirname(os.path.realpath(drts.python.__file__)) + '/assets/maps' 

def is_map_installed(self, map_name):
    return os.path.exists(DRTS_MAPS_DIR + os.sep + map_name)

def install_map(self, map_path, force=False):
    if force or not self.is_map_installed(DRTS_MAPS_DIR, os.path.basename(map_path)):
        if not force: rp.report("{map} is not installed, installing on DeepRTS...".format(map=os.path.basename(map_path)))
        copyfile(map_path, DRTS_MAPS_DIR + os.sep + os.path.basename(map_path))
    else:
        rp.report("{map} is already installed.".format(map=os.path.basename(map_path)))

def uninstall_map(self, map_path):
    if self.is_map_installed(DRTS_MAPS_DIR, os.path.basename(map_path)):
        os.remove(DRTS_MAPS_DIR + os.sep + os.path.basename(map_path))
        rp.report("{map} was removed.".format(map=os.path.basename(map_path)))
    else:
        rp.report("{map} is not installed.".format(map=os.path.basename(map_path)))
        
def show_available_maps(self):
    rp.report('Available maps on DeepRTS:')
    rp.report(os.listdir(DRTS_MAPS_DIR))

def  
