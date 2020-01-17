from .commands import DeepRTSMapView 

class RunnerBuilder():

    COMMANDS = [DeepRTSMapView]

    @staticmethod
    def build(parser, args):
        runner = None

        for cls in RunnerBuilder.COMMANDS:
            if (args.command == cls.COMMAND):
                runner = cls(parser, args)
                break

        return runner
