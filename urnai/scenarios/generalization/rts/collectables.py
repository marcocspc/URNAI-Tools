import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0,parentdir) 

from scenario.base.abscenario import ABScenario

class GeneralizedCollectablesScenario(ABScenario):

    DEEP_RTS_GAME = 0

    def __init__(self, game = GeneralizedCollectablesScenario.DEEP_RTS_GAME)
