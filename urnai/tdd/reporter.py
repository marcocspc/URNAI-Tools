class Reporter():

    '''
        This class should be used to print instead of print()
        It should be imported as follows:
        from tdd.reporter import Reporter as rp
        And then the function report should be called to print:
        rp.report("My message")
        If the message is a debug one, a level different from 0
        should be used:
        rp.report("Debug message", 2)
    '''

    #0 = default, any message
    VERBOSITY_LEVEL = 0

    @staticmethod
    def report(message, verbosity_lvl = 0):
        if (verbosity_lvl <= Reporter.VERBOSITY_LEVEL):
            print(message)

    @staticmethod
    def set_verbosity(level):
        Reporter.VERBOSITY_LEVEL = level

    @staticmethod
    def get_verbosity():
        return Reporter.VERBOSITY_LEVEL
