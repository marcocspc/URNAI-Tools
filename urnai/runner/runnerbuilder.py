#from urnai.runner.commands import DeepRTSRunner, TrainerRunner, SC2Runner
from urnai.runner.commands import TrainerRunner, SC2Runner

class RunnerBuilder():

    #COMMANDS = [DeepRTSRunner, TrainerRunner, SC2Runner]
    COMMANDS = [TrainerRunner, SC2Runner]

    @staticmethod
    def build(parser, args):
        runner = None

        for cls in RunnerBuilder.COMMANDS:
            if args.command == cls.COMMAND:
                runner = cls(parser, args)
                break

        return runner
