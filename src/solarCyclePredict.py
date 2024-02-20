# Perform Long-term Solar Cycle Prediction

# ----------------------------------------------------------------------------------------------------------------------
# Top-level Imports:
import numpy as np
from dtw import *
from scipy.signal import find_peaks
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
# Local Imports:
from tools import solarToolbox

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
# Helper function(s):
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

def makeTimeAxis(data, units='years'):
    """
    Given some 1D time series data, make a corresponding time array with the desired base units.
    :param data:
    :param units:
        A string denoting the time cadence of the axis to be returned. Valid arguments are: 'seconds', 'minutes',
        'hours', 'days', 'months' (corresponds to 30 days), and 'years'. Default is 'years'. NOTE: YOU MUST KNOW THE
        NATIVE TIME RESOLUTION OF THE INPUT DATA TO USE THIS CORRECTLY. THIS FUNCTION ASSUMES THAT THE NATIVE RESOLUTION
        OF THE DATA IS MONTHLY.
    :return timeAxis:
        The time axis in the desired units.
    """
    timeAxis = np.linspace(0, len(data)-1, len(data))
    if units == 'years':
        factor = 12.
    elif units == 'months':
        factor = 1.
    elif units == 'days':
        factor = 1/30.
    elif units == 'hours':
        factor = 1/720.
    elif units == 'minutes':
        factor = 1/43200.
    elif units == 'seconds':
        factor = 1/2592000.
    else:
        raise ValueError("Invalid arguement ")
    timeAxis = timeAxis/factor
    return timeAxis

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
    cycleAreas = np.asarray(cycleAreas)

    # FIGURE: Solar Cycle Progression, with identified peaks and troughs:
    plt.figure(figsize=(18, 9))
    # plt.plot(dailyTimes, dailySpots)
    # plt.plot(monthlyTimes, monthlySpots)
    # plt.plot(smoothedTimes, smoothedSpots)
    plt.plot(clippedDailyTimes, clippedModifiedDailySpots, label='Daily SSN')
    plt.plot(clippedMonthlyTimes, clippedMonthlySpots, label='Monthly SSN')
    plt.plot(clippedSmoothedTimes, clippedSmoothedSpots, label='13-Month Smoothed SNN')
    for i in range(len(cyclePeaks)):
        plt.axvline(x=smoothedTimes[cyclePeaks[i]], color='k')
    for j in range(len(cycleTroughs)):
        plt.axvline(x=smoothedTimes[cycleTroughs[j]], color='r')
    # Axes/labels/saving:
    plt.xlabel('Date')
    plt.ylabel('SSN')
    plt.title('Solar Cycle Progression', fontsize=18)
    plt.legend(loc='best', framealpha=1)
    plt.savefig(figures_directory+'solarCycleProgression.png', dpi=300)

    # Extract each solar cycle for the purpose of Superposed Epoch Analysis and Dynamic Time Warping:
    boundaries = [solarToolbox.find_nearest(np.asarray(clippedMonthlyTimes), element)[0] for element in np.asarray(smoothedTimes)[cycleTroughs]]
    superposedMonthlySSN = solarToolbox.superpose(clippedMonthlySpots, boundaries)
    superposedSmoothedSSN = solarToolbox.superpose(clippedSmoothedSpots, boundaries)
    fig, axs = plt.subplots(figsize=(14, 7), nrows=1, ncols=2, sharey=True, sharex=True)
    for element in superposedMonthlySSN:
        axs[0].plot(makeTimeAxis(element), element)
    axs[0].set_xlabel('Years')
    axs[0].set_ylabel('Monthly-Averaged SSN')
    for element in superposedSmoothedSSN:
        axs[1].plot(makeTimeAxis(element), element)
    axs[1].set_xlabel('Years')
    axs[1].set_ylabel('13-Month Smoothed SSN')
    fig.suptitle('Overlaid Solar Cycles', fontsize=18)
    plt.savefig(figures_directory+'overlaidSolarCycles.png', dpi=300)

    # Perform standard Superposed Epoch Analysis, using the mean cycle duration as the normalized epoch timeline:
    normalizedSuperposedMonthlySSN = solarToolbox.SEA(superposedMonthlySSN)
    meanNSM_SSN = np.mean(normalizedSuperposedMonthlySSN, axis=0)
    fig, axs = plt.subplots(figsize=(14, 7), nrows=1, ncols=2, sharey=True, sharex=True)
    axs[0].plot(makeTimeAxis(meanNSM_SSN), normalizedSuperposedMonthlySSN.T)
    axs[0].plot(makeTimeAxis(meanNSM_SSN), meanNSM_SSN, color='k', linewidth=5)
    axs[0].set_xlabel('Years')
    axs[0].set_ylabel('Monthly-Averaged SSN')
    normalizedSuperposedSmoothedSSN = solarToolbox.SEA(superposedSmoothedSSN)
    meanNSS_SSN = np.mean(normalizedSuperposedSmoothedSSN, axis=0)
    axs[1].plot(makeTimeAxis(meanNSS_SSN), normalizedSuperposedSmoothedSSN.T)
    axs[1].plot(makeTimeAxis(meanNSS_SSN), meanNSS_SSN, color='k', linewidth=5)
    axs[1].set_xlabel('Years')
    axs[1].set_ylabel('13-Month Smoothed SSN')
    fig.suptitle('Superposed Solar Cycles (Method 1)', fontsize=18)
    plt.savefig(figures_directory + 'superposedSolarCyclesMethod1.png', dpi=300)

    # Perform CUSTOM Superposed Epoch Analysis, using the mean peak location, mean e-folding time, AND mean cycle duration to construct the normalized epoch timeline (analagous to Katus, et al. 2015: https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2012JA017915):
    customNormedSuperMonthlySNN = solarToolbox.customSEA(superposedMonthlySSN)
    meanCustomNSM_SSN = np.mean(customNormedSuperMonthlySNN, axis=0)
    fig, axs = plt.subplots(figsize=(14, 7), nrows=1, ncols=2, sharey=True, sharex=True)
    axs[0].plot(makeTimeAxis(meanCustomNSM_SSN), customNormedSuperMonthlySNN.T)
    axs[0].plot(makeTimeAxis(meanCustomNSM_SSN), meanCustomNSM_SSN, color='k', linewidth=5)
    axs[0].set_xlabel('Years')
    axs[0].set_ylabel('Monthly-Averaged SSN')
    customNormedSuperSmoothedSNN = solarToolbox.customSEA(superposedSmoothedSSN)
    meanCustomNSS_SSN = np.mean(customNormedSuperSmoothedSNN, axis=0)
    axs[1].plot(makeTimeAxis(meanCustomNSS_SSN), customNormedSuperSmoothedSNN.T)
    axs[1].plot(makeTimeAxis(meanCustomNSS_SSN), meanCustomNSS_SSN, color='k', linewidth=5)
    axs[1].set_xlabel('Years')
    axs[1].set_ylabel('13-Month Smoothed SSN')
    fig.suptitle('Superposed Solar Cycles (Method 2)', fontsize=18)
    plt.savefig(figures_directory + 'superposedSolarCyclesMethod2.png', dpi=300)

    # Perform the Dynamic Time Warping for each succesive cycle:
    def warp(cycle_b, cycle_a):
        # Citation:
        # T. Giorgino. Computing and Visualizing Dynamic Time Warping Alignments in R: The dtw Package.
        # J. Stat. Soft., doi:10.18637/jss.v031.i07.
        alignment = dtw(cycle_b, cycle_a, keep_internals=True)
        # Display the warping curve, i.e. the alignment curve
        # alignment.plot(type="threeway")
        # Align and plot with the Rabiner-Juang type VI-c unsmoothed recursion
        # dtw(cycle_b, cycle_a, keep_internals=True,
        #     step_pattern=rabinerJuangStepPattern(6, "c")) \
        #     .plot(type="twoway", offset=-2)
        ## See the recursion relation, as formula and diagram
        print(rabinerJuangStepPattern(6, "c"))
        # rabinerJuangStepPattern(6, "c").plot()
        return alignment

    alignments = []
    for i in range(customNormedSuperSmoothedSNN.shape[0]-1):
        cycle_A = customNormedSuperMonthlySNN[i]
        cycle_B = customNormedSuperMonthlySNN[i+1]
        alignments.append(warp(cycle_B, cycle_A))

    # ----------------------------------------------------------------------------------------------------------------------
    # TODO: MODEL THE BEHAVIOR OF SOLAR CYCLE PARAMETERS (start by relating various parameters to others)
    # 0: Observing the Gnevyshev-Ohl Rule:
    cycleSums = np.array([np.sum(element) for element in superposedMonthlySSN])
    xAxis = np.linspace(1, len(cycleSums), len(cycleSums))
    oddCycles = cycleSums[::2]
    xAxisOdd = xAxis[::2]
    evenCycles = cycleSums[1::2]
    xAxisEven = xAxis[1::2]
    cyclePairs = [(w,x,y,z) for w,x,y,z in zip(evenCycles[0:-1], oddCycles[1:], xAxisEven[0:-1], xAxisOdd[1:])]
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
    plt.savefig(figures_directory+'gnevyshev-ohl.png', dpi=300)

    # 1: Peak SC Amplitude vs. SC Duration
    popt, pcov = curve_fit(solarToolbox.linear, cycleLengths, monthlyPeaksVals)
    pred_monthly = solarToolbox.linear(cycleLengths[np.argsort(cycleLengths)], *popt)
    r2_monthly = r2_score(monthlyPeaksVals, pred_monthly)
    popt2, pcov2 = curve_fit(solarToolbox.quadratic, cycleLengths, monthlyPeaksVals)
    pred_monthly2 = solarToolbox.quadratic(cycleLengths[np.argsort(cycleLengths)], *popt2)
    r2_monthly2 = r2_score(monthlyPeaksVals, pred_monthly2)
    #
    popt3, pcov3 = curve_fit(solarToolbox.linear, cycleLengths, smoothedPeaksVals)
    pred_smoothed = solarToolbox.linear(cycleLengths[np.argsort(cycleLengths)], *popt3)
    r2_smoothed_1 = r2_score(monthlyPeaksVals, pred_smoothed)
    popt4, pcov4 = curve_fit(solarToolbox.quadratic, cycleLengths, smoothedPeaksVals)
    pred_smoothed2 = solarToolbox.quadratic(cycleLengths[np.argsort(cycleLengths)], *popt4)
    r2_smoothed_2 = r2_score(monthlyPeaksVals, pred_smoothed2)
    #
    fig, axs = plt.subplots(figsize=(16, 7), nrows=1, ncols=2, sharey=True, sharex=True)
    axs[0].scatter(cycleLengths, monthlyPeaksVals)
    axs[0].plot(cycleLengths[np.argsort(cycleLengths)], pred_monthly, color='r', label=r'Linear Fit ($R^2\approx'+str(np.round(r2_monthly, 2))+'$)')
    axs[0].plot(cycleLengths[np.argsort(cycleLengths)], pred_monthly2, color='m', label=r'Quadratic Fit ($R^2\approx'+str(np.round(r2_monthly2, 2))+'$)')
    axs[0].set_ylabel('Maximum Amplitude (SSN)')
    axs[0].set_xlabel('Duration (years)')
    axs[0].set_title('SC Maximum Amplitude vs. SC Duration (monthly-averaged)')
    axs[0].legend(loc='best')
    axs[1].scatter(cycleLengths, smoothedPeaksVals)
    axs[1].plot(cycleLengths[np.argsort(cycleLengths)], pred_smoothed, color='r', label=r'Linear Fit ($R^2\approx'+str(np.round(r2_smoothed_1, 2))+'$)')
    axs[1].plot(cycleLengths[np.argsort(cycleLengths)], pred_smoothed2, color='m', label=r'Quadratic Fit ($R^2\approx'+str(np.round(r2_smoothed_2, 2))+'$)')
    axs[1].set_ylabel('Maximum Amplitude (SSN)')
    axs[1].set_xlabel('Duration (years)')
    axs[1].set_title('SC Maximum Amplitude vs. SC Duration (13-month Averaged)')
    axs[1].legend(loc='best')
    plt.savefig(figures_directory+'maxAmplitude_vs_duration.png', dpi=300)

    # TODO: 2: SC Amplitude vs. Area Under the Curve
    solarToolbox.plotData(cycleAreas, monthlyPeaksVals, figname=figures_directory + 'maxAmplitude_vs_area_under_curve.png',
                          figStrings=['Area-Under-the-Curve', 'Maximum Amplitude (SSN)',
                                      'SC Maximum Amplitude vs. Area-Under-the-Curve (monthly-averaged)'])

    # TODO: 3: SC Duration vs. Area Under the Curve
    solarToolbox.plotData(cycleLengths, cycleAreas, figname=figures_directory + 'area_under_curve_vs_cycleLength.png',
                          figStrings=['Cycle Duration (years)', 'Area-Under-the-Curve',
                                      'SC Duration vs. Area-Under-the-Curve (monthly-averaged)'])

    # TODO: SC Amplitude vs. e-folding Time
    cycleEFoldingTimes = np.array([solarToolbox.eFold(element)[0] for element in superposedMonthlySSN])/12. # In units of years


    # TODO: SC Duration vs. e-folding Time

    # TODO: Area Under the Curve vs. e-folding Time

    # ----------------------------------------------------------------------------------------------------------------------
    # TODO: Extrapolate SC parameters to future cycles (compare with simple parametric modeling).
    # Method 1:
    #    a: Estimate future solar cycle by extrapolating with the DTW
    #    b: Require that the Gnevyshev-Ohl Rule is adhered to (determine if it is valid to apply - e.g. when monthly SSN > 125 and cycle e-folding decay time > 6 years)

    # TODO: Use Dynamic Time Warping to assess how well the predictions worked (except those for SC25's latter half, SC26, and SC27)

    sys.exit(0)
# ----------------------------------------------------------------------------------------------------------------------
