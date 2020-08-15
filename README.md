# URNAI-Tools
URNAI Tools is a modular Deep Reinforcement Learning (DRL) toolkit that supports multiple environments, such as [PySC2](https://github.com/deepmind/pysc2), [OpenAI Gym](https://github.com/openai/gym), [ViZDoom](https://github.com/mwydmuch/ViZDoom) and [DeepRTS](https://github.com/cair/deep-rts). The main goal of URNAI Tools is to provide an easy-to-use modular platform for the development of DRL agents, so that developers can reuse as much code as possible whenever they create new agents. Each part of the DRL cycle, such as the action space, state representation, reward function, algorithm etc, is considered a module in URNAI, which allows researchers to swap any of those modules as they wish. To supply that need, URNAI comes with a series of out-of-the-box DRL algorithms, environment wrappers, action wrappers, reward functions, state representations etc, allowing developers to easily assemble different learning configurations and quickly iterate through them.

## Getting Started

Follow these instructions to get a working copy of the project on your PC. It's a good idea to use the 'solve_x.py' files as a base to start developing your own agents.

### Prerequisites

- Python 3
- Python 3 PIP

### Basic installation

- You can install from pypi:
**WARNING**, you cannot install extras from this option because pypi doesn't allow dependencies from github repositories. If you need a full installation, go to next section.
```
pip3 install urnai
```

- Or you can install from this repository:
```
pip3 install git+https://github.com/pvnetto/URNAI-Tools/ 
```

The basic installation will install all the *basic* required dependencies, including OpenAI Gym and SC2LE. But for other supported environments, you will need to install them for yourself. We describe how to do this on the next section. 

To use tensorflowp-cpu instead of gpu, go to Optional below.

### Optional

#### Starcraft II

SC2LE is already marked as dependency, so it will be automatically installed by Pip. But you need to install Starcraft II and download the mini-games and maps to Starcraft II, to do this, you can head to:

[How to install Starcraft II and Maps](https://github.com/deepmind/pysc2#get-starcraft-ii) 

#### VizDoom

Before setting urnai to install vizdoom, please see if you have all dependencies installed.

Go [here](https://github.com/mwydmuch/ViZDoom/blob/master/doc/Building.md#deps) first.

To install urnai with vizdoom support, use:

- On Unix:
```
URNAI_VIZDOOM=1 pip3 install git+https://github.com/marcocspc/URNAI-Tools/ 
```

- On Windows:
```
set "URNAI_VIZDOOM=1" && pip3 install git+https://github.com/marcocspc/URNAI-Tools/
```

#### DeepRTS 

To install urnai with DeepRTS support, use:

- On Unix:
```
URNAI_DEEPRTS=1 pip3 install git+https://github.com/marcocspc/URNAI-Tools/
```

- On Windows:
```
set "URNAI_DEEPRTS=1" && pip3 install git+https://github.com/marcocspc/URNAI-Tools/
```

#### Full Install (with all optional environments)

To install urnai with all optional environments, use:

- On Unix:
```
URNAI_DEEPRTS=1 URNAI_VIZDOOM=1 URNAI_2048=1 pip3 install git+https://github.com/marcocspc/URNAI-Tools/
```

- On Windows:
```
set "URNAI_DEEPRTS=1" && set "URNAI_VIZDOOM=1" && pip3 install git+https://github.com/marcocspc/URNAI-Tools/
```

#### Tensorflow CPU

By default, urnai depends on tensorflow-gpu, to use tf-cpu instead, use:

- On Unix:
```
URNAI_TF_CPU=1 pip3 install git+https://github.com/marcocspc/URNAI-Tools/ 
```

- On Windows:
```
set "URNAI_TF_CPU=1" && pip3 install git+https://github.com/marcocspc/URNAI-Tools/
```

#### Latest Dependencies 

By default, URNAI fixes all dependencies' versions. If you need to install those dependencies in theirs latest versions, use: 

- On Unix:
```
URNAI_LATEST_DEPS=1 pip3 install git+https://github.com/marcocspc/URNAI-Tools/
```

- On Windows:
```
set "URNAI_LATEST_DEPS=1" && pip3 install git+https://github.com/marcocspc/URNAI-Tools/
```

### Running the examples

From version 0.0.2+ you can use json-files:

```
git clone https://github.com/marcocspc/URNAI-Tools 
cd 'URNAI-Tools/urnai/test/solves'
urnai train --json-file=solve_x.json
```

The Alternative to using JSON files is to run one of our solve_x.py files in your Python interpreter. These files are used to instantiate and train an AI agent.
There are a few files to choose from. To see the solve files for games and scenarios that we have already solved (achieved a reasonable level of success) check the [solves directory](https://github.com/marcocspc/URNAI-Tools/tree/master/urnai/solves). To see the solve files that are currently being worked on, and that we haven't yet found a successful solving strategy, check out the [test/solves directory](https://github.com/marcocspc/URNAI-Tools/tree/master/urnai/test/solves).

## Command line

You can now use urnai on command line. Commands:

To see what you can do, use:
```
urnai -h
```

## Building your own code

Follow these instructions to start developing new stuff using our library.

### Building an agent for a supported environment

- Building a PySC2 agent
- Building an OpenAI Gym agent
- Building a PyGame Learning Environment agent

### Building a new DRL model

To build a new model, you should check the readme in the [urnai.models](https://github.com/pvnetto/URNAI-Tools/tree/master/urnai/models) module.

### Integrating a new environment

To integrate a new environment, you might want to check the readme in the [urnai.envs](https://github.com/pvnetto/URNAI-Tools/tree/master/urnai/envs) module.

## Roadmap

Here you'll find all the things that we plan to do in this project. **Bold** items are work in progress.

* [ ] Documentation
  * [ ] **Core codebase documentation/commenting**
  * [ ] **Documentation for each of the main modules (env, agents, models etc)**
  * [ ] **Statistics for solved problems**
* [ ] Support for new environments
  * [ ] [Arcade Learning Environment](https://github.com/mgbellemare/Arcade-Learning-Environment)
  * [ ] [Gym Retro](https://github.com/openai/retro)
  * [X] [Vizdoom](https://github.com/mwydmuch/ViZDoom)
  * [X] [DeepRTS](https://github.com/cair/deep-rts)
* [ ] More Deep RL algorithms
  * [X] Policy Gradient
  * [X] Double Deep Q-Learning
  * [ ] **A3C**
  * [ ] Curiosity-Driven Learning
  * [ ] Proximal Policy Optimization
* [X] Core codebase improvements
  * [X] Logger class
  * [X] Save model parameters
  * [X] Persistance of training parameters (saving / loading)
  * [X] Automatic generation of training graphs (avg. reward, avg. win rate etc.)
* [ ] Solve more problems
  * [X] Frozenlake
  * [X] Cartpole-V0
  * [X] Cartpole-V1
  * [X] StarCraft II - Simple 64 Map - Very Easy difficulty
  * [ ] **StarCraft II - Simple 64 Map - Easy Difficulty**


## Authors

* **Francisco de Paiva Marques Netto** - *Initial work* - [pvnetto](https://github.com/pvnetto)
* **Luiz Paulo de Carvalho Alves** - *Integration and experimentation with StarCraft II* - [lpdcalves](https://github.com/lpdcalves)
