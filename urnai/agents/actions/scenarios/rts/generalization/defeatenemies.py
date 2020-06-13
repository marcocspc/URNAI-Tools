from urnai.agents.actions import sc2 as scaux
from .findanddefeat import FindAndDefeatDeepRTSActionWrapper, FindAndDefeatStarcraftIIActionWrapper 
from pysc2.lib import actions, features, units
from statistics import mean

class DefeatEnemiesDeepRTSActionWrapper(FindAndDefeatDeepRTSActionWrapper):
    pass

class DefeatEnemiesStarcraftIIActionWrapper(FindAndDefeatStarcraftIIActionWrapper):
    def __init__(self):
        super().__init__()

        self.ver_threshold = 2
        self.hor_threshold = 2

    pass
