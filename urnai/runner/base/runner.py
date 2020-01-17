import abc

class Runner(abc.ABC):

    def __init__(self, parser, args):
        self.parser = parser
        self.args = args 

    @abc.abstractmethod
    def run(self):
        pass
