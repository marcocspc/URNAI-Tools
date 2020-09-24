import numpy as np
from os.path import expanduser, sep
import time

def save_iterable_as_csv(iterable, file_name="urnai_iterable_" + time.strftime("%Y%m%d_%H%M%S"), directory=expanduser("~"), convert_to_int = False, convert_to_string = False, delimiter=','):
    '''
        Saves any iterable supported by numpy (such as List)
        as a csv file. It converts all values into int by default.
    '''

    if ".csv" not in file_name: file_name += ".csv"

    csv = None
    if convert_to_int: 
        csv = np.array(iterable).astype(int) 
        np.savetxt(directory + sep + file_name, csv, fmt='%i',delimiter=delimiter)
    elif convert_to_string:
    else:
        csv = np.array(iterable)
        np.savetxt(directory + sep + file_name, csv, delimiter=delimiter)
