from .base.runner import Runner

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

        if (self.args.map is not None): 
            print("Starting DeepRTS using map " + self.args.map)
            stamp = os.stat(self.args.map).st_mtime 
            drts = DeepRTSEnv(render=True,map=self.args.map)
            drts.reset()

            try:
                while True:
                    current_stamp = os.stat(self.args.map).st_mtime 
                    if current_stamp != stamp:
                        stamp = current_stamp
                        drts.stop()
                        drts = DeepRTSEnv(render=True,map=self.args.map)
                        drts.reset()
            except KeyboardInterrupt:
                print("Bye!")
        else:
            self.parser.error("--map was not informed.")
        
