import abc

class Runner(abc.ABC):

    def __init__(self, args):
        self.args = args 

    @abc.abstractmethod
    def run(self):
        pass
