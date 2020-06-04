from urnai.utils.module_specialist import get_cls
from .trainer import Trainer
import json
import os

class JSONTrainer(Trainer):

    def __init__(self, json_path):
        self.pickle_black_list = []
        with open(json_path, "r") as json_file:
            self.trainings = json.loads(json_file.read())

    def start_training(self):
        for training in self.trainings:
            env_cls = get_cls("urnai", "envs", training["env"]["class"])
            env = env_cls(**training["env"]["params"])

            action_wrapper_cls = get_cls("urnai.agents", "actions", training["action_wrapper"]["class"])
            action_wrapper = action_wrapper_cls(**training["action_wrapper"]["params"])

            state_builder_cls = get_cls("urnai.agents", "states", training["state_builder"]["class"])
            state_builder = state_builder_cls(**training["state_builder"]["params"])

            reward_cls = get_cls("urnai.agents", "rewards", training["reward"]["class"])
            reward = reward_cls(**training["reward"]["params"])

            model_cls = get_cls("urnai", "models", training["model"]["class"])
            model = model_cls(action_wrapper=action_wrapper, state_builder=state_builder, **training["model"]["params"])

            agent_cls = get_cls("urnai", "agents", training["agent"]["class"])
            agent = agent_cls(model, reward, **training["agent"]["params"])

            self.setup(env, agent, **training["trainer"]["params"])

            try:
                self.train(**training["json_trainer"]["train"])
            except KeyError as ke:
                if 'train' in str(ke): pass

            try:
                self.play(**training["json_trainer"]["play"])
            except KeyError as ke:
                if 'play' in str(ke): pass

    def save_extra(self, save_path):
        super().save_extra(save_path)

        json_path = save_path + os.path.sep + "training_params.json"
        with open(json_path, "w+") as out_file:
            out_file.write(json.dumps(self.trainings, indent=4))
