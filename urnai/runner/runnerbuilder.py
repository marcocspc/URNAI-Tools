from .commands import DeepRTSRunner, TrainerRunner

class RunnerBuilder():

    COMMANDS = [DeepRTSRunner, TrainerRunner]

    @staticmethod
    def build(parser, args):
        runner = None

        for cls in RunnerBuilder.COMMANDS:
            if args.command == cls.COMMAND:
                runner = cls(parser, args)
                break

        return runner
