# URNAI-Tools
URNAI Tools is a modular Deep Reinforcement Learning library that supports multiple environments, such as [PySC2](https://github.com/deepmind/pysc2), [OpenAI Gym](https://github.com/openai/gym), and [PyGame Learning Environment](https://github.com/ntasfi/PyGame-Learning-Environment). The main goal of URNAI Tools is to provide an easy way to develop DRL agents in a way that allows the developer to reuse as much code as possible when developing different agents, and that also allows him to reuse previously implemented models in different environments and to integrate new environments easily when necessary. The main advantage of using URNAI Tools is that the models you make for one environment will also work in other environments, so you can prototype new agents for different environments very easily.

## Getting Started

Follow these instructions to get a working copy of the project on your PC. It's a good idea to use the 'solve_x.py' files as a base to start developing your own agents.

### Prerequisites

- Python 3
- Python 3 PIP

### Basic installation

- You can install from pypi:
```
pip3 install urnai
```

- Or you can install from this repository:
```
pip3 install git+https://github.com/pvnetto/URNAI-Tools/ 
```

The basic installation will install all the *basic* required dependencies, including OpenAI Gym and SC2LE. But for other supported environments, you will need to install them for yourself. We describe how to do this on the next section. 

### Optional

#### Starcraft II

SC2LE is already marked as dependency, so it will be automatically installed by Pip. But you need to install Starcraft II and download the mini-games and maps to Starcraft II, to do this, you can head to:

[How to install Starcraft II and Maps](https://github.com/deepmind/pysc2#get-starcraft-ii) 

#### 2048 Support

To install urnai with gym-2048 support, use:

- On Unix:
```
URNAI_2048=1 pip3 install urnai 
```

- On Windows:
```
set "URNAI_2048=1" && pip3 install urnai 
```

#### VizDoom

To install urnai with vizdoom support, use:

- On Unix:
```
URNAI_VIZDOOM=1 pip3 install urnai 
```

- On Windows:
```
set "URNAI_VIZDOOM=1" && pip3 install urnai 
```

#### DeepRTS 

To install urnai with DeepRTS support, use:

- On Unix:
```
URNAI_DEEPRTS=1 pip3 install urnai 
```

- On Windows:
```
set "URNAI_DEEPRTS=1" && pip3 install urnai 
```

#### Full Install (with all optional environments)

To install urnai with all optional environments, use:

- On Unix:
```
URNAI_DEEPRTS=1 URNAI_VIZDOOM=1 URNAI_2048=1 pip3 install urnai 
```

- On Windows:
```
set "URNAI_DEEPRTS=1" && set "URNAI_VIZDOOM=1" && set "URNAI_2048=1" && pip3 install urnai 
```

### Running the examples

To execute any of the examples we've included, just navigate to the project's folder and run them using Python.

```
cd 'project/save/path'
python solve_x.py
```

## Guide

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
  * [ ] Core codebase documentation/commenting
  * [ ] **Documentation for each of the main modules (env, agents, models etc)**
  * [ ] Statistics for solved problems
* [ ] Support for new environments
  * [ ] [Arcade Learning Environment](https://github.com/mgbellemare/Arcade-Learning-Environment)
  * [ ] [Gym Retro](https://github.com/openai/retro)
  * [X] [Vizdoom](https://github.com/mwydmuch/ViZDoom)
* [ ] More Deep RL algorithms
  * [X] Policy Gradient
  * [ ] **A3C**
  * [ ] Curiosity-Driven Learning
  * [ ] Proximal Policy Optimization
* [X] Core codebase improvements
  * [X] Logger class
  * [ ] Save model parameters (currently saves only the weights)
* [ ] Solve more problems
  * [X] Frozenlake
  * [X] Cartpole-V0
  * [X] Cartpole-V1
  * [X] Taxi-V2
  * [ ] Flappy Bird
  * [ ] **StarCraft II - Simple 64 Map - Very Easy difficulty**



## Authors

* **Francisco de Paiva Marques Netto** - *Initial work* - [pvnetto](https://github.com/pvnetto)
