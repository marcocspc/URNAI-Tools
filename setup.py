from setuptools import setup

setup(
    name = "urnai",
    packages = ["urnai"],
    install_requires = [
        ''
        ],
    entry_points = {
        "console_scripts": ['urnai=urnai.urnai_cmd:main']
        },
    version = "0.0.1",
    description = "A modular Deep Reinforcement Learning library that supports multiple environments, such as PySC2, OpenAI Gym, and PyGame Learning Environment.",
    long_description = "URNAI Tools is a modular Deep Reinforcement Learning library that supports multiple environments, such as PySC2, OpenAI Gym, and PyGame Learning Environment. The main goal of URNAI Tools is to provide an easy way to develop DRL agents in a way that allows the developer to reuse as much code as possible when developing different agents, and that also allows him to reuse previously implemented models in different environments and to integrate new environments easily when necessary. The main advantage of using URNAI Tools is that the models you make for one environment will also work in other environments, so you can prototype new agents for different environments very easily.",
    author = "UFRN-IMD-URNAITeam",
    author_email = "urnaiteam@gmail.com",
    url = "https://github.com/pvnetto/URNAI-Tools",
)
