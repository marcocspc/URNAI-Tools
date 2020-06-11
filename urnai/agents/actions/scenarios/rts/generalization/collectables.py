from urnai.agents.actions.base.abwrapper import ActionWrapper
from urnai.agents.actions import sc2 as scaux
from pysc2.lib import actions, features, units
from statistics import mean
from pysc2.env import sc2_env

class CollectablesDeepRTSActionWrapper(ActionWrapper):
    def __init__(self):
        self.move_number = 0

        self.previousunit = 0 
        self.nextunit = 1
        self.moveleft = 2
        self.moveright = 3
        self.moveup = 4
        self.movedown = 5
        self.moveupleft = 6
        self.moveupright = 7
        self.movedownleft = 8
        self.movedownright = 9
        self.attack = 10
        self.harvest = 11
        self.build0 = 12
        self.build1 = 13
        self.build2 = 14
        self.noaction = 15

        self.actions = [self.previousunit, self.nextunit, self.moveleft, self.moveright, self.moveup, self.movedown,
                self.moveupleft, self.moveupright, self.movedownleft, self.movedownright, self.attack, self.harvest,
                self.build0, self.build1, self.build2, self.noaction] 

        self.excluded_actions = [self.previousunit, self.nextunit, 
                self.moveupleft, self.moveupright, self.movedownleft, 
                self.movedownright, self.attack, self.harvest,
                self.build0, self.build1, self.build2, self.noaction] 

        self.final_actions = list(set(self.actions) - set(self.excluded_actions))

    def is_action_done(self):
        return True

    def reset(self):
        self.move_number = 0

    def get_actions(self):
        return self.final_actions
    
    def get_excluded_actions(self, obs):        
        return []

    def get_action(self, action_idx, obs):
        return self.final_actions[action_idx]

class CollectablesStarcraftIIActionWrapper(ActionWrapper):

    def __init__(self):
        self.move_number = 0

        self.hor_threshold = 2
        self.ver_threshold = 2

        self.moveleft = 0
        self.moveright = 1
        self.moveup = 2
        self.movedown = 3

        self.actions = [self.moveleft, self.moveright, self.moveup, self.movedown] 
        self.pending_actions = []

    def is_action_done(self):
        return True

    def reset(self):
        self.move_number = 0

    def get_actions(self):
        return self.actions
    
    def get_excluded_actions(self, obs):        
        return []

    def get_action(self, action_idx, obs):
        if len(self.pending_actions) > 0:
            return [self.pending_actions.pop()]
        else:
            self.solve_action(action_idx, obs)
            return [actions.RAW_FUNCTIONS.no_op()]

    def solve_action(self, action_idx, obs):
        if action_idx == self.moveleft:
            self.move_left(obs)
        elif action_idx == self.moveright:
            self.move_right(obs)
        elif action_idx == self.moveup:
            self.move_up(obs)
        elif action_idx == self.movedown:
            self.move_down(obs)
    
    def move_left(self, obs):
        army = scaux.select_army(obs, sc2_env.Race.terran)
        xs = [unit.x for unit in army]
        ys = [unit.y for unit in army]

        new_army_x = int(mean(xs)) - self.hor_threshold 
        new_army_y = int(mean(ys))

        for unit in army:
            self.pending_actions.append(actions.RAW_FUNCTIONS.Move_pt("now", unit.tag, [new_army_x, new_army_y]))
            
    def move_right(self, obs):
        army = scaux.select_army(obs, sc2_env.Race.terran)
        xs = [unit.x for unit in army]
        ys = [unit.y for unit in army]

        new_army_x = int(mean(xs)) + self.hor_threshold 
        new_army_y = int(mean(ys))

        for unit in army:
            self.pending_actions.append(actions.RAW_FUNCTIONS.Move_pt("now", unit.tag, [new_army_x, new_army_y]))

    def move_down(self, obs):
        army = scaux.select_army(obs, sc2_env.Race.terran)
        xs = [unit.x for unit in army]
        ys = [unit.y for unit in army]

        new_army_x = int(mean(xs))
        new_army_y = int(mean(ys)) + self.ver_threshold 

        for unit in army:
            self.pending_actions.append(actions.RAW_FUNCTIONS.Move_pt("now", unit.tag, [new_army_x, new_army_y]))

    def move_up(self, obs):
        army = scaux.select_army(obs, sc2_env.Race.terran)
        xs = [unit.x for unit in army]
        ys = [unit.y for unit in army]

        new_army_x = int(mean(xs))
        new_army_y = int(mean(ys)) - self.ver_threshold 

        for unit in army:
            self.pending_actions.append(actions.RAW_FUNCTIONS.Move_pt("now", unit.tag, [new_army_x, new_army_y]))

