# This module contains miscellaneous tools for loading in and manipulating data for solar cycle prediction

# ----------------------------------------------------------------------------------------------------------------------
# Top-level imports:
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use('Qt5Agg')
from scipy.interpolate import InterpolatedUnivariateSpline
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

def find_nearest(array, value):
    """
    Find the index and value the item in array closest to a supplied value.
    :param: array: ndarray
        The array to search over.
    :param: value:
        The value to compare to all of the values of the array.
    :return: idx: int
        The index of the element in array closest to value.
    :return: array[idx]: float or int
        The quantity itself corresponding to idx.
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

def superpose(data, boundaries):
        """
        Given a stream of data and boundaries in that data, extract the individual phenomena and superpose them.
        :param data: arraylike
            A 1d array or list.
        :param boundaries: arraylike
            A 1d array or list of boundaries for each phenomenon to be superposed. Each value in this argument should be
            an index.
        :return superposedData:
            The individual superposed phenomena.
        """
        superposedData = []
        # Collect the phenomena:
        for i in range(len(boundaries)-1):
            superposedData.append(data[boundaries[i]:boundaries[i+1]])
        return superposedData

def SEA(superposedPhenomena):
    """
    Given an output from 'superpose', perform normalization to warp the phenomena to the normalized timeline taken
    as the mean cycle duration.
    :param superposedPhenomena: list
        Output from superpose.
    :return normalizedSuperposedPhenomena: ndarray
        The superposed phenomena conformed to the normalized timeline.
    """
    epochDuration = int(np.floor(np.mean([len(element) for element in superposedPhenomena])))
    normalizedTimeline = np.linspace(0, epochDuration, epochDuration)
    normalizedSuperposedPhenomena = []
    for element in superposedPhenomena:
        xSample = np.linspace(0, epochDuration, len(element))
        currentSpline = InterpolatedUnivariateSpline(xSample, element)
        currentNormalizedPhenomena = currentSpline(normalizedTimeline)
        normalizedSuperposedPhenomena.append(currentNormalizedPhenomena)
    normalizedSuperposedPhenomena = np.asarray(normalizedSuperposedPhenomena)
    return normalizedSuperposedPhenomena
# ----------------------------------------------------------------------------------------------------------------------



