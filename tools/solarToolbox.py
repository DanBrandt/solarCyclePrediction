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

def eFold(data):
    """
    Return the index of the e-folding time of the data. Assumes that the e-folding time is calculated after the
    maximum of the data.
    :param data: arraylike
        A 1D array of data.
    :return eFoldingTime: float
        The e-folding time in units of indices.
    :return eFoldingIndex: int
        The actual index of where the e-folding time occurs
    :return eFoldingValue: float
        The value of the data at the e-folding time.
    """
    maxLoc = np.argmax(data)
    dataSubset = np.asarray(data)[maxLoc:]
    eFoldingActualValue = np.asarray(data)[maxLoc] / np.e
    eFoldingIndex, eFoldingValue = find_nearest(dataSubset, eFoldingActualValue)
    eFoldingTime = eFoldingIndex + maxLoc
    return eFoldingTime, eFoldingIndex, eFoldingValue

def customSEA(superposedPhenomena):
    """
    Given an output from 'superpose', perform normalization to warp the phenomena to the normalized timeline taken
    as constructed from the following epoch markers: mean peak-location, mean e-folding time, and mean cycle
    duration.
    :param superposedPhenomena: list
        Output from superpose.
    :return normalizedSuperposedPhenomena: ndarray
        The superposed phenomena conformed to the normalized timeline.
    """
    # Obtain peak locations:
    maxLocs = []
    for element in superposedPhenomena:
        maxLocs.append(np.argmax(element))
    meanMaxLoc = int(np.round(np.mean(maxLocs)))
    # Obtain peak e-folding times (locations):
    eFoldingTimes = []
    eFoldingIndices = []
    for element in superposedPhenomena:
        result = eFold(element)
        eFoldingTimes.append(result[0])
        eFoldingIndices.append(result[1])
    meanEFoldTime = int(np.round(np.mean(eFoldingTimes)))
    meanEFoldIndex = int(np.round(np.mean(eFoldingIndices)))
    # Obtain mean cycle duration:
    decayTimes = []
    i = 0
    for element in superposedPhenomena:
        decayTimes.append( len(element) - eFoldingTimes[i] )
        i += 1
    meanEpochDuration = int(np.floor(np.mean([len(element) for element in superposedPhenomena])))
    meanDecayTime = int(np.round(np.mean(decayTimes)))

    # Sanity check:
    # import matplotlib.pyplot as plt
    # import matplotlib
    # matplotlib.use('Qt5Agg')
    # plt.figure()
    # plt.plot(superposedPhenomena[0])
    # plt.axvline(x=eFoldingTimes[0])

    # Construct the normalize epoch timeline from the epoch markers, piece-by-piece:
    start_to_max = np.linspace(0, meanMaxLoc-1, meanMaxLoc)
    max_to_e_fold = np.linspace(meanMaxLoc, meanEFoldTime-1, meanEFoldIndex)
    e_fold_to_end = np.linspace(meanEFoldTime, meanEpochDuration-1, meanDecayTime)

    # Helper function for resampling along each section of the normalized epoch:
    def impose(original_data, new_duration, new_axis):
        xSample = np.linspace(new_axis[0], new_axis[-1], len(original_data))
        spline = InterpolatedUnivariateSpline(xSample, original_data)
        xSampleNew = np.linspace(new_axis[0], new_axis[-1], new_duration)
        return spline(xSampleNew)

    # Superpose all of the phenomena onto the normalized timeline, stitching each section together:
    normalizedSuperposedPhenomena = []
    j = 0
    for element in superposedPhenomena:
        original_start_to_max = element[:np.argmax(element)]
        normed_start_to_max = impose(original_start_to_max, len(start_to_max), start_to_max)
        original_max_to_e_fold = element[np.argmax(element):eFoldingTimes[j]]
        normed_max_to_e_fold = impose(original_max_to_e_fold, len(max_to_e_fold), max_to_e_fold)
        original_e_fold_to_end = element[eFoldingTimes[j]:]
        normed_e_fold_to_end = impose(original_e_fold_to_end, len(e_fold_to_end), e_fold_to_end)
        currentNormalizedPhenomena = np.concatenate((normed_start_to_max, normed_max_to_e_fold, normed_e_fold_to_end))
        normalizedSuperposedPhenomena.append(currentNormalizedPhenomena)
        j += 1

    normalizedSuperposedPhenomena = np.asarray(normalizedSuperposedPhenomena)
    return normalizedSuperposedPhenomena

def linear(x, a, b):
    """
    Sample linear function.
    :param x: arraylike
        1D data.
    :param a: float
        Coefficient (slope).
    :param b: float
        Coefficient (intercept).
    :return y: float or arraylike
        Dependent variable.
    """
    return a*x + b

def quadratic(x, a, b, c):
    """
    Sample quadratic function.
    :param x: arraylike
        1D data.
    :param a: float
        First coefficient.
    :param b: float
        Second coefficient.
    :param c: float
        Third coefficient.
    :return y: float or arraylike
        Dependent variable.
    """
    return a*x**2 + b*x + c

def sinusoid(x, a, b, c, d):
    """
    Sample sinuisoidal function.
    :param x: arraylike
        1D data.
    :param a: float
        First coefficient.
    :param b: float
        Second coefficient.
    :param c: float
        Third coefficient.
    :param d: float
        Fourth coefficient.
    :return y: float or arraylike
        Dependent variable.
    """
    return a*np.sin(b*x + c) + d
# ----------------------------------------------------------------------------------------------------------------------



