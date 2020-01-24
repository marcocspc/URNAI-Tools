import argparse
from .runnerbuilder import RunnerBuilder  
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from utils.error import *

class ParserBuilder():

    DESCRIPTION = "A modular Deep Reinforcement Learning library that supports multiple environments, such as PySC2, OpenAI Gym, and PyGame Learning Environment."

    @staticmethod
    def DefaultParser():
        parser = argparse.ArgumentParser(description=ParserBuilder.DESCRIPTION)
        avail_pos_cmd = []
        avail_opt_cmd = []

        for cls in RunnerBuilder.COMMANDS:
            avail_pos_cmd += cls.COMMANDS
            avail_opt_cmd += cls.OPT_COMMANDS

        if not (ParserBuilder.check_unique_entries(avail_pos_cmd) and ParserBuilder.check_unique_entries(avail_opt_cmd)):
            raise CommandsNotUniqueError("There are repeated positional commands or optional commands on commands.py")

        for cmd in avail_pos_cmd:
            parser.add_argument(cmd['command'], help=cmd['help'], type=cmd['type'])

        for cmd in avail_opt_cmd:
            parser.add_argument(cmd['command'], help=cmd['help'], type=cmd['type'])

        return parser

    @staticmethod
    def check_unique_entries(cmd_list):
        new_cmd_list = []

        for cmd in cmd_list:
            new_cmd_list.append(cmd['command'])

        return len(new_cmd_list) == len(set(new_cmd_list))
