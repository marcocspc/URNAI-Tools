from setuptools import setup, find_packages
import os

git_url = '{package} @ git+https://github.com/{user}/{package}.git/@{version}#egg={package}-0'

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
        ],
    extras_require = {
        '2048' : ['gym-2048', git_url.format(user='ntasfi', package='PyGame-Learning-Environment', version='master')],
        'vizdoom' : ['vizdoom'],
        'deeprts' : [git_url.format(user='UIA-CAIR', package='DeepRTS', version='e54dc6c')],
        },
    entry_points = {
        "console_scripts": ['urnai=urnai.urnai_cmd:main']
        },
    version = "0.0.1-2",
    description = "A modular Deep Reinforcement Learning library that supports multiple environments, such as PySC2, OpenAI Gym, and PyGame Learning Environment.",
    long_description = "URNAI Tools is a modular Deep Reinforcement Learning library that supports multiple environments, such as PySC2, OpenAI Gym, and PyGame Learning Environment. The main goal of URNAI Tools is to provide an easy way to develop DRL agents in a way that allows the developer to reuse as much code as possible when developing different agents, and that also allows him to reuse previously implemented models in different environments and to integrate new environments easily when necessary. The main advantage of using URNAI Tools is that the models you make for one environment will also work in other environments, so you can prototype new agents for different environments very easily.",
    author = "UFRN-IMD-URNAITeam",
    author_email = "urnaiteam@gmail.com",
    url = "https://github.com/pvnetto/URNAI-Tools",
)
