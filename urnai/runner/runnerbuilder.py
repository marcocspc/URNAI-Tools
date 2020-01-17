from .commands import DeepRTSMapView 

class RunnerBuilder():

    COMMANDS = [DeepRTSMapView]

    @staticmethod
    def build(parser):
        args = parser.parse_args()
        runner = None

        for cls in COMMANDS:
            if (args.command == cls.COMMAND):
                runner = cls(args)

        return runner
