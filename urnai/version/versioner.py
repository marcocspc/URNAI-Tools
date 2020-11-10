from urnai.base.savable import Savable
from tdd.reporter import Reporter as rp
import os
import subprocess

class Versioner(Savable):

    VERSION = "1.0-43f5281"

    def __init__(self):
        self.__curr_version = self.get_current_version()
        self.version = self.__curr_version
        self.pickle_black_list.append("__curr_version") 

    def get_current_version(self):
        my_dir = os.path.dirname(os.path.realpath(__file__))
        #since urnai git repo is two levels above
        #this file, I have to get the parent folder twice
        parent_dir = os.path.dirname(my_dir)
        git_dir = os.path.dirname(parent_dir)

        try:
            branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=git_dir)
            git_hash = subprocess.check_output(['git', 'rev-parse', '--short', branch], cwd=git_dir)

            return "{}-{}".format(branch, git_hash)
        except CalledProcessError:
            return Versioner.VERSION

    def load_extra(self, persist_path):
        self.__curr_version = self.get_current_version()

    def ask_for_continue(self):
        if self.version != self.__curr_version:
            answer = ""
            while answer.lower() != "y" and answer.lower() != "n":
                answer = rp.input("The loaded training version is {} and the current version is {}. This difference can cause some kind of error while proceeding to the training, do you wish to continue? [y/n]".format(self.version, self.__curr_version), "n")

                if answer.lower() == "n":
                    rp.report("The training was stopped.")
                    exit()
