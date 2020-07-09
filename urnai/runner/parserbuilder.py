import argparse
from urnai.runner.runnerbuilder import RunnerBuilder  
import os,sys,inspect
from urnai.utils.error import *

class ParserBuilder():

    DESCRIPTION = "A modular Deep Reinforcement Learning library that supports multiple environments, such as PySC2, OpenAI Gym, and PyGame Learning Environment."

    @staticmethod
    def DefaultParser():
        parser = argparse.ArgumentParser(description=ParserBuilder.DESCRIPTION)
        avail_pos_cmd = []
        avail_opt_cmd = []

        for cls in RunnerBuilder.COMMANDS:
            avail_pos_cmd.append(cls.COMMAND)
            avail_opt_cmd += cls.OPT_COMMANDS

        opt_cmd_aux_list = []

        for cmd in avail_opt_cmd:
            opt_cmd_aux_list.append(cmd['command'])

        if not (ParserBuilder.check_unique_entries(avail_pos_cmd) and ParserBuilder.check_unique_entries(opt_cmd_aux_list)):
            raise CommandsNotUniqueError("There are repeated positional commands or optional commands on commands.py")

        parser.add_argument('command', help='Which command urnai should run. Choices: {%(choices)s}', choices=avail_pos_cmd, metavar='COMMAND')

        for cmd in avail_opt_cmd:
            #parser.add_argument(cmd['command'], help=cmd['help'], action=cmd['action'], type=cmd['type'], metavar=cmd['metavar'])
            parser.add_argument(cmd['command'], **{key:cmd[key] for key in cmd if key != 'command'})

        return parser

    @staticmethod
    def check_unique_entries(lst):
        return len(lst) == len(set(lst))
