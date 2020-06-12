from setuptools import setup, find_packages
import os

def is_optional_enabled(optional):
    return os.environ.get(optional, None) is not None

# Dependencies versions
# VERSION_TF = "==2.2.0"
# VERSION_VIZDOOM = "==1.1.7"
# VERSION_DRTS = "stable"
# VERSION_PLE = "3dbe79d"
# VERSION_2048 = "==0.2.6"
# VERSION_ABSL = "==0.9.0"
# VERSION_GYM = "==0.10.11"
# VERSION_NUMPY = "==1.18.4"
# VERSION_MATPLOTLIB = "==3.2.0"
# VERSION_KERAS = "==2.3.1"
# VERSION_PYSC2 = "==3.0.0"
# VERSION_PANDAS = "==1.0.1"

VERSION_TF = ""
VERSION_VIZDOOM = ""
VERSION_DRTS = "stable"
VERSION_PLE = ""
VERSION_2048 = ""
VERSION_ABSL = ""
VERSION_GYM = ""
VERSION_NUMPY = ""
VERSION_MATPLOTLIB = ""
VERSION_KERAS = ""
VERSION_PYSC2 = ""
VERSION_PANDAS = ""


VIZDOOM = 'URNAI_VIZDOOM'
TF_CPU = 'URNAI_TF_CPU'
G2048 = 'URNAI_2048'
DEEPRTS = 'URNAI_DEEPRTS'

git_url = '{package} @ git+https://github.com/{user}/{repo}@{branch}#egg={package}'
dep_links = []
dep_list = []
tf = 'tensorflow-gpu' + VERSION_TF

if is_optional_enabled(DEEPRTS):
    print("DeepRTS installation enabled.")
    dep_list.append(git_url.format(user='marcocspc', repo='deep-rts', branch=VERSION_DRTS, package='deeprts'))

if is_optional_enabled(VIZDOOM):
    print("VizDoom installation enabled.")
    dep_list.append('vizdoom' + VERSION_VIZDOOM)

if is_optional_enabled(G2048):
    print("Gym-2048 installation enabled.")
    dep_list.append(git_url.format(user='ntasfi', package='ple', branch=VERSION_PLE, repo='PyGame-Learning-Environment'))
    dep_list.append('gym-2048' + VERSION_2048)

if is_optional_enabled(TF_CPU):
    print("Tensorflow cpu will be installed instead of Tensorflow GPU.")
    tf = 'tensorflow' + VERSION_TF

setup(
    name = "urnai",
    packages = find_packages(),
    install_requires = [
        'absl-py' + VERSION_ABSL,
        'gym' + VERSION_GYM,
        tf,
        'numpy' + VERSION_NUMPY,
        'matplotlib' + VERSION_MATPLOTLIB,
        'keras' + VERSION_KERAS,
        'pysc2' + VERSION_PYSC2,
        'pandas' + VERSION_PANDAS,
        ] + dep_list,
    dependency_links=dep_links,
    entry_points = {
        "console_scripts": ['urnai=urnai.urnai_cmd:main']
        },
    version = "0.0.2-1",
    description = "A modular Deep Reinforcement Learning library that supports multiple environments, such as PySC2, OpenAI Gym, and PyGame Learning Environment.",
    long_description = "URNAI Tools is a modular Deep Reinforcement Learning library that supports multiple environments, such as PySC2, OpenAI Gym, and PyGame Learning Environment. The main goal of URNAI Tools is to provide an easy way to develop DRL agents in a way that allows the developer to reuse as much code as possible when developing different agents, and that also allows him to reuse previously implemented models in different environments and to integrate new environments easily when necessary. The main advantage of using URNAI Tools is that the models you make for one environment will also work in other environments, so you can prototype new agents for different environments very easily.",
    author = "UFRN-IMD-URNAITeam",
    author_email = "urnaiteam@gmail.com",
    url = "https://github.com/marcocspc/URNAI-Tools",
)
