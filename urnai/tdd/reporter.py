import os
import pickle
from datetime import datetime

class Reporter():

    '''
        This class should be used to print instead of print()
        It should be imported as follows:
        from urnai.tdd.reporter import Reporter as rp
        or
        from tdd.reporter import Reporter as rp
        And then the function report should be called to print:
        rp.report("My message")
        If the message is a debug one, a level different from 0
        should be used:
        rp.report("Debug message", 2)
    '''

    #0 = default, any message
    VERBOSITY_LEVEL = 0
    MESSAGES = []

    @staticmethod
    def report(message, verbosity_lvl = 0, end = "\n"):
        date = "[URNAI REPORT AT " + str(datetime.now()) + "] "
        message = date + message
        if (verbosity_lvl <= Reporter.VERBOSITY_LEVEL):
            print(message, end=end)
            Reporter.MESSAGES.append(message)

    @staticmethod
    def save(persist_path):
        pickle_path = persist_path + os.path.sep + "report.pkl"
        with open(pickle_path, "wb") as pickle_out: 
            pickle.dump(Reporter.MESSAGES, pickle_out)

        string_out = ""
        for line in Reporter.MESSAGES:
            string_out += line + "\n"

        with open(pickle_path.replace(".pkl", ".txt"), "w") as text_out: 
            text_out.write(string_out)

    @staticmethod
    def load(persist_path):
        pickle_path = persist_path + os.path.sep + "report.pkl"
        exists_pickle = os.path.isfile(pickle_path)
        if exists_pickle:
            with open(pickle_path, "rb") as pickle_in: 
                Reporter.MESSAGES = pickle.load(pickle_in)
