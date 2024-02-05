# This module contains miscellaneous tools for loading in and manipulating data for solar cycle prediction

# ----------------------------------------------------------------------------------------------------------------------
# Top-level imports:
import numpy as np
from datetime import datetime
import sys
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# Functions
def readSILSO(filename, daily=False):
    """
    Given a filename of sunspot data from SILSO, read it in and grab the sunspot time series data.
    Data source should be: https://www.sidc.be/SILSO/datafiles
    :param filename: str
        The name or location of the file.
    :param daily: bool
        Indicates whether or not the file being read in corresponds to daily data or not. Default is False.
    :return: times: ndarray
        An array of datetimes for each sunspot value.
    :return: spots: ndarray
        An array of sunspot numbers.
    """
    with open(filename, "r") as solarFile:
        contents = solarFile.readlines()
        times = []
        spots = np.zeros(len(contents))
        i = 0
        for line in contents:
            parsed = line.split()
            if daily == True:
                timeVal = datetime(int(parsed[0]), int(parsed[1]), int(parsed[2]), 12) # Put all observations at noon of each day
            else:
                timeVal = datetime(int(parsed[0]), int(parsed[1]), 12) # Put all observations at noon of each day
            times.append(timeVal)
            if parsed[3] == -1:
                spots[i] = np.nan
            else:
                if daily == True:
                    spots[i] = float(parsed[4])
                else:
                    spots[i] = float(parsed[3])
            i += 1
    return times, spots

# ----------------------------------------------------------------------------------------------------------------------



