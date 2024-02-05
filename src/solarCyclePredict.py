# Perform Long-term Solar Cycle Prediction

# ----------------------------------------------------------------------------------------------------------------------
# Top-level Imports:
import numpy as np
import sys
from scipy.signal import find_peaks
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
# Local Imports:
from tools import solarToolbox
# ----------------------------------------------------------------------------------------------------------------------

def clip(timeData, boundaries):
    """
    Given some list of datetimes and two dates, return only the data between those two dates, along with the indices
    corresponding to data between those two dates.
    :param timeData: list
        A list of datetimes.
    :param boundaries: list
        A two-element list where the first element is the starting date, and the second element is the ending date. The
        starting date must be older than the ending date.
    :return clippedTimes: list
        The datetimes between the boundaries.
    :return inds: list
        A list of indices of the time data between the boundaries.
    """
    inds = np.where((np.asarray(timeData) >= boundaries[0]) & (np.asarray(timeData) <= boundaries[-1]))[0]
    clippedTimes = np.asarray(timeData)[inds]
    return clippedTimes, inds

# ----------------------------------------------------------------------------------------------------------------------
# Execution
if __name__ == '__main__':
    # 1: LOAD IN DATA:
    dailyTimes, dailySpots = solarToolbox.readSILSO('../data/SN_d_tot_V2.0.txt', daily=True)
    monthlyTimes, monthlySpots = solarToolbox.readSILSO('../data/SN_m_tot_V2.0.txt')
    smoothedTimes, smoothedSpots = solarToolbox.readSILSO('../data/SN_ms_tot_V2.0.txt')

    # 2: EXTRACT SOLAR CYCLE PARAMETERS:
    # Find the peaks and terminators of each cycle, by looking at each one individually:
    cyclePeaks, _ = find_peaks(smoothedSpots, distance=85, prominence=5)
    cycleTroughs, _ = find_peaks(-smoothedSpots, distance=85)
    # Clip the data so that the very first cycle is ignored, and Solar Cycle 25 is not considered:
    bounds = [smoothedTimes[cycleTroughs[0]]], smoothedTimes[cycleTroughs[-1]]
    clippedDailyTimes, inds = clip(dailyTimes, bounds)
    clippedDailySpots = dailySpots[inds]
    clippedModifiedDailySpots = np.where(clippedDailySpots==-1, np.nan, clippedDailySpots)
    clippedMonthlyTimes, inds = clip(monthlyTimes, bounds)
    clippedMonthlySpots = monthlySpots[inds]
    clippedSmoothedTimes, inds = clip(smoothedTimes, bounds)
    clippedSmoothedSpots = smoothedSpots[inds]
    clippedCyclePeaks = cyclePeaks[1:-1]
    peaksTimes = np.asarray(monthlyTimes)[clippedCyclePeaks]
    monthlyPeaksVals = np.asarray(monthlySpots)[clippedCyclePeaks]
    smoothedPeaksVals = np.asarray(smoothedSpots)[clippedCyclePeaks]
    # Length of each solar cycle:
    cycleLengths = []
    for i in range(len(cycleTroughs)-1):
        cycleLengths.append( (smoothedTimes[cycleTroughs[i+1]] - smoothedTimes[cycleTroughs[i]]).days )
    cycleLengths = np.array([element/365 for element in cycleLengths])
    # Ascending and descending times for each solar cycle:
    leftCycleTroughs = cycleTroughs[:-1]
    rightCycleTroughs = cycleTroughs[1:]
    cycleAscendingTimes = []
    cycleAscendingValsMonthly = []
    cycleAscendingValsSmoothed = []
    cycleDescendingTimes = []
    cycleDescendingValsMonthly = []
    cycleDescendingValsSmoothed = []
    for i in range(len(cycleLengths)):
        # Ascending:
        cycleAscendingTimes.append( (peaksTimes[i] - smoothedTimes[leftCycleTroughs[i]]).days )
        cycleAscendingValsMonthly.append( monthlyPeaksVals[i] - monthlySpots[leftCycleTroughs[i]] )
        cycleAscendingValsSmoothed.append( smoothedPeaksVals[i] - smoothedSpots[leftCycleTroughs[i]] )
        # Descending:
        cycleDescendingTimes.append( (smoothedTimes[rightCycleTroughs[i]] - peaksTimes[i]).days )
        cycleDescendingValsMonthly.append( monthlySpots[rightCycleTroughs[i]] - monthlyPeaksVals[i] )
        cycleDescendingValsSmoothed.append( smoothedSpots[rightCycleTroughs[i]] - smoothedPeaksVals[i] )
    # Rates of Ascent and Descent:
    ascentRatesMonthly = np.array([x/y for x,y in zip(cycleAscendingValsMonthly, cycleAscendingTimes)])
    ascentRatesSmoothed = np.array([x/y for x,y in zip(cycleAscendingValsSmoothed, cycleAscendingTimes)])
    descentRatesMonthly = np.array([x/y for x,y in zip(cycleDescendingValsMonthly, cycleDescendingTimes)])
    descentRatesSmoothed = np.array([x/y for x,y in zip(cycleDescendingValsSmoothed, cycleDescendingTimes)])
    # Ratios of Ascent to Descent:
    ratiosMonthly = np.array([x/y for x,y in zip(cycleAscendingValsMonthly, cycleDescendingValsMonthly)])
    ratiosSmoothed = np.array([x / y for x, y in zip(cycleAscendingValsSmoothed, cycleDescendingValsSmoothed)])
    # Area-under-the-Curve for each solar cycle:
    cycleAreas = []
    for i in range(len(cycleLengths)):
        yVals = monthlySpots[cycleTroughs[i]:cycleTroughs[i+1]]
        cycleAreas.append( trapezoid(yVals) )

    plt.figure()
    # plt.plot(dailyTimes, dailySpots)
    # plt.plot(monthlyTimes, monthlySpots)
    # plt.plot(smoothedTimes, smoothedSpots)
    plt.plot(clippedDailyTimes, clippedModifiedDailySpots)
    plt.plot(clippedMonthlyTimes, clippedMonthlySpots)
    plt.plot(clippedSmoothedTimes, clippedSmoothedSpots)
    for i in range(len(cyclePeaks)):
        plt.axvline(x=smoothedTimes[cyclePeaks[i]], color='k')
    for j in range(len(cycleTroughs)):
        plt.axvline(x=smoothedTimes[cycleTroughs[j]], color='r')

    # TODO: 3: MODEL THE BEHAVIOR OF SOLAR CYCLE PARAMETERS


    # TODO: Extrapolate them to future cycles (compare with simple parametric modeling).

    # TODO: Use Dynamic Time Warping to assess how well the predictions worked (except those for SC25's latter half,
    # SC26, and SC27)

    sys.exit(0)
# ----------------------------------------------------------------------------------------------------------------------
