from .findanddefeat import FindAndDefeatGeneralizedStatebuilder 
from urnai.utils.constants import RTSGeneralization, Games 
from pysc2.lib import units as sc2units
import urnai.agents.actions.sc2 as sc2aux 

class DefeatEnemiesGeneralizedStatebuilder(FindAndDefeatGeneralizedStatebuilder):

    def __init__(self, method=RTSGeneralization.STATE_MAP):
        super().__init__(method=method)
