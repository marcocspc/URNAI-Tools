class Reporter():

    '''
        This class should be used to print instead of print()
        It should be imported as follows:
        from tdd.Reporter import report
        And then the function report should be called to print:
        report("My message")
        If the message is a debug one, a level different from 0
        should be used:
        report(2, "Debug message")

    #0 = default, any message
    VERBOSITY_LEVEL = 0

    @staticmethod
    def report(verbosity_lvl = 0, message):
        if (verbosity_lvl <= VERBOSITY_LEVEL):
            print(message)

    @staticmethod
    def set_verbosity(level):
        VERBOSITY_LEVEL = level

    @staticmethod
    def get_verbosity():
        return VERBOSITY_LEVEL
