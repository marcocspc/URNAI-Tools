from .base.abwrapper import ActionWrapper


class VizdoomWrapper(ActionWrapper):

    def __init__(self, env):
        self.move_number = 0

        attack = [1, 0, 0, 0, 0, 0, 0, 0]
        use = [0, 1, 0, 0, 0, 0, 0, 0]
        move_forward = [0, 0, 1, 0, 0, 0, 0, 0]
        move_backward = [0, 0, 0, 1, 0, 0, 0, 0]
        move_right = [0, 0, 0, 0, 1, 0, 0, 0]
        move_left = [0, 0, 0, 0, 0, 1, 0, 0]
        turn_right = [0, 0, 0, 0, 0, 0, 1, 0]
        turn_left = [0, 0, 0, 0, 0, 0, 0, 1]

        self.actions = [attack, use, move_forward, move_backward,
            move_right, move_left, turn_right, turn_left]


    def is_action_done(self):
        return True

    
    def reset(self):
        self.move_number = 0


    def get_actions(self):
        return self.actions


    def get_excluded_actions(self, obs):        
        return []


    def get_action(self, action_idx, obs):
        return self.actions[action_idx.index(1)]

class VizdoomHealthGatheringWrapper(ActionWrapper):

    def __init__(self, env):
        self.move_number = 0
        
        move_forward = [0, 0, 1, 0, 0, 0, 0, 0]
        turn_left = [0, 0, 0, 0, 0, 0, 0, 1]
        turn_right = [0, 0, 0, 0, 0, 0, 1, 0]

        self.actions = [move_forward, turn_right, turn_left]


    def is_action_done(self):
        return True

    
    def reset(self):
        self.move_number = 0


    def get_actions(self):
        return self.actions


    def get_excluded_actions(self, obs):        
        return []


    def get_action(self, action_idx, obs):
        return self.actions[self.actions.index(action_idx)]
