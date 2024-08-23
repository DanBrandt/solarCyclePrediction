# This script generates figures and performs long-term forecasting for future Solar Cycles.

# Top-level imports:
import numpy as np
from scipy.signal import find_peaks
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
import sys

# Local imports:
from tools import solarToolbox
from tools import corFoci

# ----------------------------------------------------------------------------------------------------------------------
# Global plotting settings:
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

# ----------------------------------------------------------------------------------------------------------------------
# Directory Management
figures_directory = '../results/figures/'

# ----------------------------------------------------------------------------------------------------------------------
# Execution:
if __name__ == '__main__':
    # 1 - Load in data:
    dailyTimes, dailySpots = solarToolbox.readSILSO('../data/SN_d_tot_V2.0.txt', daily=True)
    monthlyTimes, monthlySpots = solarToolbox.readSILSO('../data/SN_m_tot_V2.0.txt')
    smoothedTimes, smoothedSpots = solarToolbox.readSILSO('../data/SN_ms_tot_V2.0.txt')

    # 2 - Extract Solar Cycle Parameters:
    # Find the peaks and terminators of each cycle, by looking at each one individually:
    cyclePeaks, _ = find_peaks(smoothedSpots, distance=85, prominence=5)
    cycleTroughs, _ = find_peaks(-smoothedSpots, distance=85)
    # Clip the data so that the very first cycle is ignored, and Solar Cycle 25 is not considered:
    bounds = [smoothedTimes[cycleTroughs[0]]], smoothedTimes[cycleTroughs[-1]]
    clippedDailyTimes, inds = solarToolbox.clip(dailyTimes, bounds)
    clippedDailySpots = dailySpots[inds]
    clippedModifiedDailySpots = np.where(clippedDailySpots == -1, np.nan, clippedDailySpots)
    clippedMonthlyTimes, inds = solarToolbox.clip(monthlyTimes, bounds)
    clippedMonthlySpots = monthlySpots[inds]
    clippedSmoothedTimes, inds = solarToolbox.clip(smoothedTimes, bounds)
    clippedSmoothedSpots = smoothedSpots[inds]
    clippedCyclePeaks = cyclePeaks[1:-1]
    clippedCycleTroughs = cycleTroughs[1:-1]
    peaksTimes = np.asarray(monthlyTimes)[clippedCyclePeaks]
    monthlyPeaksVals = np.asarray(monthlySpots)[clippedCyclePeaks]
    smoothedPeaksVals = np.asarray(smoothedSpots)[clippedCyclePeaks]
    # Length of each solar cycle:
    cycleLengths = []
    for i in range(len(cycleTroughs) - 1):
        cycleLengths.append((smoothedTimes[cycleTroughs[i + 1]] - smoothedTimes[cycleTroughs[i]]).days)
    cycleLengths = np.array([element / 365 for element in cycleLengths])
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
        cycleAscendingTimes.append((peaksTimes[i] - smoothedTimes[leftCycleTroughs[i]]).days)
        cycleAscendingValsMonthly.append(monthlyPeaksVals[i] - monthlySpots[leftCycleTroughs[i]])
        cycleAscendingValsSmoothed.append(smoothedPeaksVals[i] - smoothedSpots[leftCycleTroughs[i]])
        # Descending:
        cycleDescendingTimes.append((smoothedTimes[rightCycleTroughs[i]] - peaksTimes[i]).days)
        cycleDescendingValsMonthly.append(monthlySpots[rightCycleTroughs[i]] - monthlyPeaksVals[i])
        cycleDescendingValsSmoothed.append(smoothedSpots[rightCycleTroughs[i]] - smoothedPeaksVals[i])
    # Rates of Ascent and Descent:
    ascentRatesMonthly = np.array([x / y for x, y in zip(cycleAscendingValsMonthly, cycleAscendingTimes)])
    ascentRatesSmoothed = np.array([x / y for x, y in zip(cycleAscendingValsSmoothed, cycleAscendingTimes)])
    descentRatesMonthly = np.array([x / y for x, y in zip(cycleDescendingValsMonthly, cycleDescendingTimes)])
    descentRatesSmoothed = np.array([x / y for x, y in zip(cycleDescendingValsSmoothed, cycleDescendingTimes)])
    # Ratios of Ascent to Descent:
    ratiosMonthly = np.array([x / y for x, y in zip(cycleAscendingValsMonthly, cycleDescendingValsMonthly)])
    ratiosSmoothed = np.array([x / y for x, y in zip(cycleAscendingValsSmoothed, cycleDescendingValsSmoothed)])
    # Area-under-the-Curve for each solar cycle:
    cycleAreas = []
    for i in range(len(cycleLengths)):
        yVals = monthlySpots[cycleTroughs[i]:cycleTroughs[i + 1]]
        cycleAreas.append(trapezoid(yVals))
    cycleAreas = np.asarray(cycleAreas)
    # Extract each solar cycle for the purpose of Superposed Epoch Analysis (and Dynamic Time Warping):
    boundaries = [solarToolbox.find_nearest(np.asarray(clippedMonthlyTimes), element)[0] for element in
                  np.asarray(smoothedTimes)[cycleTroughs]]
    superposedMonthlySSN = solarToolbox.superpose(clippedMonthlySpots, boundaries)
    superposedSmoothedSSN = solarToolbox.superpose(clippedSmoothedSpots, boundaries)
    fig, axs = plt.subplots(figsize=(14, 7), nrows=1, ncols=2, sharey=True, sharex=True)
    for element in superposedMonthlySSN:
        axs[0].plot(solarToolbox.makeTimeAxis(element), element)
    axs[0].set_xlabel('Years')
    axs[0].set_ylabel('Monthly-Averaged $S_{\mathrm{N}}$')
    for element in superposedSmoothedSSN:
        axs[1].plot(solarToolbox.makeTimeAxis(element), element)
    axs[1].set_xlabel('Years')
    axs[1].set_ylabel('13-Month Smoothed $S_{\mathrm{N}}$')
    fig.suptitle('Overlaid Solar Cycles', fontsize=18)
    plt.savefig(figures_directory + 'overlaidSolarCycles.png', dpi=300)
    # Perform standard Superposed Epoch Analysis, using the mean cycle duration as the normalized epoch timeline:
    normalizedSuperposedMonthlySSN = solarToolbox.SEA(superposedMonthlySSN)
    meanNSM_SSN = np.mean(normalizedSuperposedMonthlySSN, axis=0)
    fig, axs = plt.subplots(figsize=(14, 7), nrows=1, ncols=2, sharey=True, sharex=True)
    axs[0].plot(solarToolbox.makeTimeAxis(meanNSM_SSN), normalizedSuperposedMonthlySSN.T)
    axs[0].plot(solarToolbox.makeTimeAxis(meanNSM_SSN), meanNSM_SSN, color='k', linewidth=5)
    axs[0].set_xlabel('Years')
    axs[0].set_ylabel('Monthly-Averaged SSN')
    normalizedSuperposedSmoothedSSN = solarToolbox.SEA(superposedSmoothedSSN)
    meanNSS_SSN = np.mean(normalizedSuperposedSmoothedSSN, axis=0)
    axs[1].plot(solarToolbox.makeTimeAxis(meanNSS_SSN), normalizedSuperposedSmoothedSSN.T)
    axs[1].plot(solarToolbox.makeTimeAxis(meanNSS_SSN), meanNSS_SSN, color='k', linewidth=5)
    axs[1].set_xlabel('Years')
    axs[1].set_ylabel('13-Month Smoothed SSN')
    fig.suptitle('Superposed Solar Cycles (Method 1)', fontsize=18)
    plt.savefig(figures_directory + 'superposedSolarCycles.png', dpi=300)
    # Perform CUSTOM Superposed Epoch Analysis, using the mean peak location, mean e-folding time, AND mean cycle duration to construct the normalized epoch timeline (analagous to Katus, et al. 2015: https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2012JA017915):
    # customNormedSuperMonthlySNN = solarToolbox.customSEA(superposedMonthlySSN)
    # meanCustomNSM_SSN = np.mean(customNormedSuperMonthlySNN, axis=0)
    # fig, axs = plt.subplots(figsize=(14, 7), nrows=1, ncols=2, sharey=True, sharex=True)
    # axs[0].plot(solarToolbox.makeTimeAxis(meanCustomNSM_SSN), customNormedSuperMonthlySNN.T)
    # axs[0].plot(solarToolbox.makeTimeAxis(meanCustomNSM_SSN), meanCustomNSM_SSN, color='k', linewidth=5)
    # axs[0].set_xlabel('Years')
    # axs[0].set_ylabel('Monthly-Averaged SSN')
    # customNormedSuperSmoothedSNN = solarToolbox.customSEA(superposedSmoothedSSN)
    # meanCustomNSS_SSN = np.mean(customNormedSuperSmoothedSNN, axis=0)
    # axs[1].plot(solarToolbox.makeTimeAxis(meanCustomNSS_SSN), customNormedSuperSmoothedSNN.T)
    # axs[1].plot(solarToolbox.makeTimeAxis(meanCustomNSS_SSN), meanCustomNSS_SSN, color='k', linewidth=5)
    # axs[1].set_xlabel('Years')
    # axs[1].set_ylabel('13-Month Smoothed SSN')
    # fig.suptitle('Superposed Solar Cycles (Method 2)', fontsize=18)
    # plt.savefig(figures_directory + 'superposedSolarCyclesMethod2.png', dpi=300)
    # E-folding times:
    cycleEFoldingTimes = np.array(
        [solarToolbox.eFold(element)[0] for element in superposedMonthlySSN]) / 12.  # In units of years
    # Observing the Gnevyshev-Ohl Rule:
    cycleSums = np.array([np.sum(element) for element in superposedMonthlySSN])
    xAxis = np.linspace(1, len(cycleSums), len(cycleSums))
    oddCycles = cycleSums[::2]
    xAxisOdd = xAxis[::2]
    evenCycles = cycleSums[1::2]
    xAxisEven = xAxis[1::2]
    cyclePairs = [(w, x, y, z) for w, x, y, z in zip(evenCycles[0:-1], oddCycles[1:], xAxisEven[0:-1], xAxisOdd[1:])]
    plt.figure(figsize=(12, 5))
    # Plot the first solar cycle
    plt.scatter(1, cycleSums[0], facecolors='midnightblue', edgecolor='midnightblue')
    plt.scatter(1, cycleSums[0], s=100, facecolors='none', edgecolor='darkviolet')
    # Plot the pairs:
    for pair in cyclePairs:
        xVals = [pair[2], pair[3]]
        yVals = [pair[0], pair[1]]
        plt.plot(xVals, yVals, marker='o', linestyle='-', color='midnightblue')
        plt.scatter(pair[3], pair[1], s=100, facecolors='white', edgecolors='darkviolet', alpha=1)
    # Plot SC24:
    plt.scatter(24, cycleSums[0], facecolors='midnightblue', edgecolor='midnightblue')
    # Axes/Labeling/Saving:
    plt.xticks(xAxis)
    plt.xlabel('Solar Cycle Number', fontweight='bold')
    plt.ylabel('Sum of Monthly SSN', fontweight='bold')
    plt.title('Gnevyshev-Ohl Rule', fontweight='bold')
    plt.savefig(figures_directory + 'gnevyshev-ohl.png', dpi=300)

    # 3 - Figure of the Solar Cycle History, along with Solar Cycle history with boundaries (include daily, monthly,
    # and 13-month average (rolling and centered) SSN (Version 2):
    plt.figure(figsize=(18, 9))
    # plt.plot(dailyTimes, dailySpots)
    # plt.plot(monthlyTimes, monthlySpots)
    # plt.plot(smoothedTimes, smoothedSpots)
    plt.plot(clippedDailyTimes, clippedModifiedDailySpots, label=r'Daily $S_{\mathrm{N}}$')
    plt.plot(clippedMonthlyTimes, clippedMonthlySpots, label=r'Monthly $S_{\mathrm{N}}$')
    plt.plot(clippedSmoothedTimes, clippedSmoothedSpots, label=r'13-Month Smoothed $S_{\mathrm{N}}$')
    for i in range(len(cyclePeaks)):
        plt.axvline(x=smoothedTimes[cyclePeaks[i]], color='k')
    for j in range(len(cycleTroughs)):
        plt.axvline(x=smoothedTimes[cycleTroughs[j]], color='r')
    # Axes/labels/saving:
    plt.xlabel('Date')
    plt.ylabel('$S_{\mathrm{N}}$')
    plt.title('Solar Cycle Progression', fontsize=18)
    plt.legend(loc='best', framealpha=1)
    plt.savefig(figures_directory + 'solarCycleProgression.png', dpi=300)

    # 4 - Figure of Solar Cycle Parameters Under Consideration:
    # Parameters: Max Amplitude, Time to Max Amplitude, Min Amplitude, Cycle Lengths, Ascending Time, Descending Time, Ratios of Ascent to Descent, Cycle Areas, and E-folding times
    cycleNum = np.linspace(1, 24, 24)
    fig, axs = plt.subplots(nrows=9, ncols=1, sharex=True, sharey=False)
    maxAmplitudes = np.asarray(smoothedSpots)[cyclePeaks[1:-1]]
    axs[0].plot(cycleNum, maxAmplitudes)
    minAmplitudes = np.asarray(smoothedSpots)[cycleTroughs[1:]]
    axs[1].plot(cycleNum, minAmplitudes)
    axs[2].plot(cycleNum, cycleAscendingTimes)
    axs[3].plot(cycleNum, cycleLengths)
    axs[4].plot(cycleNum, ascentRatesSmoothed)
    axs[5].plot(cycleNum, descentRatesSmoothed)
    axs[6].plot(cycleNum, ratiosSmoothed)
    axs[7].plot(cycleNum, cycleAreas)
    axs[8].plot(cycleNum, cycleEFoldingTimes)
    # TODO: Label the above plot...

    # Collect the parameters into an array for FOCI:
    sc_drivers = [
        ['MaxAmplitude', cycleNum[:-1], maxAmplitudes[:-1]],
        ['MinAmplitude', cycleNum[:-1], minAmplitudes[:-1]],
        ['AscentTime', cycleNum[:-1], cycleAscendingTimes[:-1]],
        ['CycleLength', cycleNum[:-1], cycleLengths[:-1]],
        ['AscentRate', cycleNum[:-1], ascentRatesSmoothed[:-1]],
        ['DescentRate', cycleNum[:-1], descentRatesSmoothed[:-1]],
        ['AscentDescentRatio', cycleNum[:-1], ratiosSmoothed[:-1]],
        ['CycleArea', cycleNum[:-1], cycleAreas[:-1]],
        ['CycleEFoldingTime', cycleNum[:-1], cycleEFoldingTimes[:-1]]
    ]

    sc_drivers_for_forecasting = [ # 'MinAmplitude_AscentTime_CycleLength_CycleArea_CycleEFoldingTime'
        maxAmplitudes[-1],
        minAmplitudes[-1],
        cycleAscendingTimes[-1],
        cycleLengths[-1],
        ascentRatesSmoothed[-1],
        descentRatesSmoothed[-1],
        ratiosSmoothed[-1],
        cycleAreas[-1],
        cycleEFoldingTimes[-1]
    ]
    print(sc_drivers_for_forecasting)

    # 5 - Results of FOCI:
    true_data_max_amplitude = ['nextMaxAmplitude', cycleNum[1:], maxAmplitudes[1:]]
    bestModel_amplitude, bestDrivers_amplitude, stats_amplitude = corFoci.relate(sc_drivers, true_data_max_amplitude, [5, 3], sc_drivers_for_forecasting)
    true_data_max_amplitude_time = ['nextMaxAmplitudeTime', cycleNum[1:], cycleAscendingTimes[1:]]
    bestModel_amplitude_time, bestDrivers_amplitude_time, stats_amplitude_time = corFoci.relate(sc_drivers, true_data_max_amplitude_time, [5, 3], sc_drivers_for_forecasting)

    # 6 - Correlation plot between FOCI results and Solar Cycle Max Amplitude & and Solar Cycle Time and Max Amplitude:

    # 7 - Validation: Running this method for PAST cycles:

    # 8 - Forecasting results:

    sys.exit(0)