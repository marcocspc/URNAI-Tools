from envs.sc2 import SC2Env
from pysc2.env import sc2_env

sc2_races = {
    "terran": sc2_env.Race.terran,
    "protoss": sc2_env.Race.protoss,
    "zerg": sc2_env.Race.zerg,
    "random": sc2_env.Race.random}
sc2_difficulties = {
    "very_easy": sc2_env.Difficulty.very_easy,
    "easy": sc2_env.Difficulty.easy,
    "medium": sc2_env.Difficulty.medium,
    "medium_hard": sc2_env.Difficulty.medium_hard,
    "hard": sc2_env.Difficulty.hard,
    "harder": sc2_env.Difficulty.harder,
    "very_hard": sc2_env.Difficulty.very_hard,
    "cheat_vision": sc2_env.Difficulty.cheat_vision,
    "cheat_money": sc2_env.Difficulty.cheat_money,
    "cheat_insane": sc2_env.Difficulty.cheat_insane}

def get_sc2_race(sc2_race: str):
    out = sc2_races.get(sc2_race)
    if out is not None:
        return out
    else:
        raise Exception("Chosen race for StarCraft II doesn't match any known races. Try: 'terran', 'protoss', 'zerg' or 'random'")

def get_sc2_difficulty(sc2_difficulty: str):
    out = sc2_difficulties.get(sc2_difficulty)
    if out is not None:
        return out
    else:
        raise Exception("Chosen difficulty for StarCraft II doesn't match any known difficulties. Try: 'very_easy', 'easy', 'medium', 'medium_hard', 'hard', 'harder', 'very_hard', 'cheat_vision', 'cheat_money' or 'cheat_insane'")
