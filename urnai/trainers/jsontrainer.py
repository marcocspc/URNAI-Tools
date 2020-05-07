from urnai.utils.module_specialist import get_cls
import json

class JSONTrainer():

    def __init__(self, json_path):
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

            trainer_cls = get_cls("urnai", "trainers", training["trainer"]["class"])
            trainer = trainer_cls(env, agent, **training["trainer"]["params"])

            if bool(training["json_trainer"]["train"]):
                trainer.train(**training["json_trainer"]["train"])

            if bool(training["json_trainer"]["play"]):
                trainer.play(**training["json_trainer"]["play"])

