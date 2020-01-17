import argparse
from .runnerbuilder import RunnerBuilder  

class ParserBuilder():

    DESCRIPTION = "A modular Deep Reinforcement Learning library that supports multiple environments, such as PySC2, OpenAI Gym, and PyGame Learning Environment."

    @staticmethod
    def DefaultParser():
        parser = argparse.ArgumentParser(description=ParserBuilder.DESCRIPTION)
        avail_cmd = []

        for cls in RunnerBuilder.COMMANDS:
            avail_cmd.append(cls.COMMAND)
        
        parser.add_argument('command', metavar='COMMAND', help='Command to run, can be one of the following: ' + str(avail_cmd))
        parser.add_argument('--map', help='Map to use on RTS environments.')

        return parser
