from setuptools import setup, find_packages
from setuptools.command.install import install
import os

#git_url = '{package} @ git+https://github.com/{user}/{package}.git/@{version}#egg={package}-0'
git_url = 'https://github.com/{user}/{package}.git/@{version}#egg={package}-0'
dep_links = []
dep_list = []

class Optionals(install):
    user_options = install.user_options + [
        ('deeprts', None, None), 
        ('vizdoom', None, None), 
        ('g2048', None, None), 
    ]

    def initialize_options(self):
        install.initialize_options(self)
        self.deeprts = None
        self.vizdoom = None
        self.g2048 = None

    def finalize_options(self):
        install.finalize_options(self)

    def run(self):
        global someopt
        someopt = self.someopt # will be 1 or None
        if self.deeprts == 1:
            print("DeepRTS installation enabled.")
            dep_links.append(git_url.format(user='UIA-CAIR', package='DeepRTS', version='e54dc6c'))

        if self.vizdoom == 1:
            print("VizDoom installation enabled.")
            dep_list.append('vizdoom')

        if self.g2048 == 1:
            print("Gym-2048 installation enabled.")
            dep_list.append('gym-2048')
            dep_links.append(git_url.format(user='ntasfi', package='PyGame-Learning-Environment', version='master'))

        install.run(self)


setup(
    name = "urnai",
    packages = find_packages(),
    install_requires = [
        'absl-py',
        'gym',
        'tensorflow',
        'numpy',
        'matplotlib',
        'keras',
        'pysc2',
        'pandas',
        ] + dep_list,
    dependency_links=dep_links,
    entry_points = {
        "console_scripts": ['urnai=urnai.urnai_cmd:main']
        },
    version = "0.0.1-2",
    description = "A modular Deep Reinforcement Learning library that supports multiple environments, such as PySC2, OpenAI Gym, and PyGame Learning Environment.",
    long_description = "URNAI Tools is a modular Deep Reinforcement Learning library that supports multiple environments, such as PySC2, OpenAI Gym, and PyGame Learning Environment. The main goal of URNAI Tools is to provide an easy way to develop DRL agents in a way that allows the developer to reuse as much code as possible when developing different agents, and that also allows him to reuse previously implemented models in different environments and to integrate new environments easily when necessary. The main advantage of using URNAI Tools is that the models you make for one environment will also work in other environments, so you can prototype new agents for different environments very easily.",
    author = "UFRN-IMD-URNAITeam",
    author_email = "urnaiteam@gmail.com",
    url = "https://github.com/pvnetto/URNAI-Tools",
    cmdclass={
        'install': Optionals,
    },
)
