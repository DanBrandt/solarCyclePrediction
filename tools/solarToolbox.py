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

# ----------------------------------------------------------------------------------------------------------------------
# Execution
if __name__ == '__main__':
    dailyTimes, dailySpots = readSILSO('../data/SN_d_tot_V2.0.txt', daily=True)
    monthlyTimes, monthlySpots = readSILSO('../data/SN_m_tot_V2.0.txt')
    smoothedTimes, smoothedSpots = readSILSO('../data/SN_ms_tot_V2.0.txt')

    # TODO: Add and test functions for cleaning data (if needed)

    # TODO: Add and execute functions for extracting solar cycle parameters (including those computed BETWEEN cycles)

    # TODO: Model the behavior of the solar cycle parameters (see how they vary with other things)

    # TODO: Extrapolate them to future cycles (compare with simple parametric modeling).

    # TODO: Use Dynamic Time Warping to assess how well the predictions worked (except those for SC25's latter half,
    # SC26, and SC27)

    sys.exit(0)
# ----------------------------------------------------------------------------------------------------------------------



