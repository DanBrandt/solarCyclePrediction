# This script generates figures and performs long-term forecasting for future Solar Cycles.

# Top-level imports:
import numpy as np
from scipy.signal import find_peaks
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
import sys, os
from datetime import datetime, timedelta
from sklearn.metrics import r2_score

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
        [solarToolbox.eFold(element)[0] for element in superposedSmoothedSSN]) / 12.  # In units of years (formerly did this with superposedMonthlySSN)
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
        ['DescentTime', cycleNum[:-1], cycleDescendingTimes[:-1]],
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
        cycleDescendingTimes[-1],
        cycleLengths[-1],
        ascentRatesSmoothed[-1],
        descentRatesSmoothed[-1],
        ratiosSmoothed[-1],
        cycleAreas[-1],
        cycleEFoldingTimes[-1]
    ]
    print(sc_drivers_for_forecasting)

    # 5 - Results of FOCI:
    override = False
    amplitude_cache_file = '../results/amplitude_cache.pkl'
    true_data_max_amplitude = ['nextMaxAmplitude', cycleNum[1:], maxAmplitudes[1:]]
    if os.path.isfile(amplitude_cache_file) == False or override == True:
        bestModel_amplitude, bestDrivers_amplitude, preds_amplitude, preds_amplitudeCI, bestDrivers_amplitude_test  = corFoci.relate(sc_drivers, true_data_max_amplitude, [5, 3], sc_drivers_for_forecasting, lambda_range=[0.01328, 0.01329])
        cachedAmplitudeResults = {
            'bestModel_amplitude': bestModel_amplitude,
            'bestDrivers_amplitude': bestDrivers_amplitude,
            'preds_amplitude': preds_amplitude,
            'preds_amplitudeCI': preds_amplitudeCI,
            'bestDrivers_amplitude_test': bestDrivers_amplitude_test
        }
        solarToolbox.savePickle(cachedAmplitudeResults, amplitude_cache_file)
    else:
        cachedAmplitudeResults = solarToolbox.loadPickle(amplitude_cache_file)
        bestModel_amplitude = cachedAmplitudeResults['bestModel_amplitude']
        bestDrivers_amplitude = cachedAmplitudeResults['bestDrivers_amplitude']
        preds_amplitude = cachedAmplitudeResults['preds_amplitude']
        preds_amplitudeCI = cachedAmplitudeResults['preds_amplitudeCI']
        bestDrivers_amplitude_test = cachedAmplitudeResults['bestDrivers_amplitude_test']

    # TODO: Fix the fact that we cannot run FOCI immediately a second time (when results from the above ARE NOT cached).
    override = True
    amplitude_time_cache_file = '../results/amplitude_time_cache.pkl'
    true_data_max_amplitude_time = ['nextMaxAmplitudeTime', cycleNum[1:], np.asarray(cycleAscendingTimes[1:])]
    if os.path.isfile(amplitude_time_cache_file) == False or override == True:
        bestModel_amplitude_time, bestDrivers_amplitude_time, preds_amplitude_time, preds_amplitude_time_CI, bestDrivers_amplitude_time_test = corFoci.relate(sc_drivers, true_data_max_amplitude_time, [5, 3], sc_drivers_for_forecasting, lambda_range=[0.0195, 0.0200]) # [0.014432,0.014434]
        cachedAmplitudeTimeResults = {
            'bestModel_amplitude_time': bestModel_amplitude_time,
            'bestDrivers_amplitude_time': bestDrivers_amplitude_time,
            'preds_amplitude_time': preds_amplitude_time,
            'preds_amplitude_time_CI': preds_amplitude_time_CI,
            'bestDrivers_amplitude_time_test': bestDrivers_amplitude_time_test
        }
        solarToolbox.savePickle(cachedAmplitudeTimeResults, amplitude_time_cache_file)
    else:
        cachedAmplitudeTimeResults = solarToolbox.loadPickle(amplitude_time_cache_file)
        bestModel_amplitude_time = cachedAmplitudeTimeResults['bestModel_amplitude_time']
        bestDrivers_amplitude_time = cachedAmplitudeTimeResults['bestDrivers_amplitude_time']
        preds_amplitude_time = cachedAmplitudeTimeResults['preds_amplitude_time']
        preds_amplitude_time_CI = cachedAmplitudeTimeResults['preds_amplitude_time_CI']
        bestDrivers_amplitude_time_test = cachedAmplitudeTimeResults['bestDrivers_amplitude_time_test']

    # 6 - Correlation plot between FOCI results and Solar Cycle Max Amplitude & and Solar Cycle Time and Max Amplitude:
    # For Amplitude...
    X_past_amplitude = np.asarray([element[-1] for element in bestDrivers_amplitude]).T
    hindcasts_amplitude = bestModel_amplitude.predict(X_past_amplitude)
    plt.figure(figsize=(7,5))
    plt.scatter(true_data_max_amplitude[2], hindcasts_amplitude, color='b', s=60)
    plt.xlabel('$S_{\mathrm{N}}$')
    plt.ylabel('$M_A$')
    plt.axline((0, 0), slope=1, color='k', linestyle='--')
    linearA = np.poly1d(np.polyfit(true_data_max_amplitude[2], hindcasts_amplitude, 1))
    R2_A = r2_score(true_data_max_amplitude[2], hindcasts_amplitude)
    x_extended_A = np.linspace(0, np.max(true_data_max_amplitude[2])+50, 100)
    plt.plot(x_extended_A, linearA(x_extended_A), color='r', label='$R^2$='+str(np.round(R2_A,4))) #(np.unique(true_data_max_amplitude[2]), linearA(np.unique(true_data_max_amplitude[2])), color='r')
    plt.title('$A_{\mathrm{max}}$: SC2 through SC24')
    plt.xlim([x_extended_A[0], x_extended_A[-1]])
    plt.legend(loc='best')
    plt.savefig(figures_directory + '/amplitude_corr.png', dpi=300)

    # For Time...
    X_past_amplitude_time = np.asarray([element[-1] for element in bestDrivers_amplitude_time]).T
    hindcasts_amplitude_time = bestModel_amplitude_time.predict(X_past_amplitude_time)
    plt.figure(figsize=(7, 5))
    plt.scatter(true_data_max_amplitude_time[2], hindcasts_amplitude_time, color='b', s=60)
    plt.xlabel('$S_{\mathrm{N}}$')
    plt.ylabel(r'$M_{\tau}$')
    plt.axline((0, 0), slope=1, color='k', linestyle='--')
    linearTau = np.poly1d(np.polyfit(true_data_max_amplitude_time[2], hindcasts_amplitude_time, 1))
    R2_tau = r2_score(true_data_max_amplitude_time[2], hindcasts_amplitude_time)
    x_extended_tau = np.linspace(0, np.max(true_data_max_amplitude_time[2]) + 50, 100)
    plt.plot(x_extended_tau, linearTau(x_extended_tau), color='r', label='$R^2$=' + str(np.round(R2_tau,
                                                                                           4)))  # (np.unique(true_data_max_amplitude[2]), linearA(np.unique(true_data_max_amplitude[2])), color='r')
    plt.title(r'$\tau_{\mathrm{max}}$: SC2 through SC24')
    plt.xlim([x_extended_tau[0], x_extended_tau[-1]])
    plt.legend(loc='best')
    plt.savefig(figures_directory + '/amplitude_time_corr.png', dpi=300)

    # 7 - Partial Dependence Plots:
    # For Amplitude...
    fig, axs = plt.subplots(1, 3)
    fig.set_size_inches(16, 6)
    titles = ['$f_{1, A}$', '$f_{2, A}$', '$f_{3, A}$']
    xlabels = ['$X_{A, 1}$', '$X_{A, 2}$', '$X_{A, 3}$']
    for i, ax in enumerate(axs):
        XX = bestModel_amplitude.generate_X_grid(term=i)
        ax.plot(XX[:, i], bestModel_amplitude.partial_dependence(term=i, X=XX))
        ax.plot(XX[:, i], bestModel_amplitude.partial_dependence(term=i, X=XX, width=.95)[1], c='r', ls='--')
        ax.set_xlabel(xlabels[i])
        ax.set_title(titles[i])
        if i == 0:
            ax.set_ylabel('Contribution to $A_{\mathrm{max}}$')
    fig.savefig(figures_directory+'/PDP_amplitude.png', dpi=300)

    # For Time...
    fig, axs = plt.subplots(1, 3)
    fig.set_size_inches(16, 6)
    titles = [r'$f_{1, \tau}$', r'$f_{2, \tau}$', r'$f_{3, \tau}$']
    xlabels = [r'$X_{\tau, 1}$', r'$X_{\tau, 2}$', r'$X_{\tau, 3}$']
    for i, ax in enumerate(axs):
        XX = bestModel_amplitude_time.generate_X_grid(term=i)
        ax.plot(XX[:, i], bestModel_amplitude_time.partial_dependence(term=i, X=XX))
        ax.plot(XX[:, i], bestModel_amplitude_time.partial_dependence(term=i, X=XX, width=.95)[1], c='r', ls='--')
        ax.set_xlabel(xlabels[i])
        ax.set_title(titles[i])
        if i == 0:
            ax.set_ylabel(r'Contribution to $\tau_{\mathrm{max}}$')
    fig.savefig(figures_directory + '/PDP_amplitude_time.png', dpi=300)

    # 8 - Validation: Running this method for PAST cycles:

    # 9 - Hindcasting/Forecasting results (w/ 68% CI):
    plt.figure(figsize=(18, 9))
    plt.plot(smoothedTimes[clippedCycleTroughs[-2]:][:-6], smoothedSpots[clippedCycleTroughs[-2]:][:-6], label=r'13-Month Smoothed $S_{\mathrm{N}}$')
    plt.axvline(x=smoothedTimes[cyclePeaks[-2]], color='grey')
    plt.axvline(x=smoothedTimes[cycleTroughs[-2]], color='r')
    plt.axvline(x=smoothedTimes[cycleTroughs[-1]], color='r')
    # Hindcast:
    xerr_24 = timedelta(days=np.abs(preds_amplitude_time[0] - preds_amplitude_time_CI[0][0]))
    yerr_24 = preds_amplitude[0] - preds_amplitudeCI[0][0]
    plt.errorbar(smoothedTimes[cycleTroughs[-2]] + timedelta(days=preds_amplitude_time[0]), preds_amplitude[0], xerr=xerr_24,
                 yerr=yerr_24, capsize=5, color='tab:orange', label='SC24 Hindcast', fmt='o')
    # Forecast
    xerr_25 = timedelta(days=np.abs(preds_amplitude_time[1] - preds_amplitude_time_CI[1][0]))
    yerr_25 = preds_amplitude[1] - preds_amplitudeCI[1][0]
    plt.errorbar(smoothedTimes[cycleTroughs[-1]] + timedelta(days=preds_amplitude_time[0]), preds_amplitude[1], xerr=xerr_25,
                 yerr=yerr_25, capsize=5, color='tab:green', label='SC25 Forecast', fmt='o')
    # Other forecasts:
    asymmetric_yerr_noaa_nasa = np.array([[20, 15]]).T # https://www.weather.gov/news/190504-sun-activity-in-solar-cycle
    plt.errorbar(datetime(2025,1,1) + timedelta(days=213.), 115., xerr=timedelta(days=244, seconds=47520), yerr=asymmetric_yerr_noaa_nasa, label='NOAA/NASA', fmt='ko', capsize=5) # 2025.583 +- 0.67
    plt.errorbar(datetime(2024,1,1) + timedelta(days=228.), 149.5, xerr=timedelta(days=25, seconds=79488), yerr=11.5, label='WDC-SILSO', fmt='bo', capsize=5) # https://www.sidc.be/article/solar-cycle-25-maximum 2024.625+-0.208
    asymmetric_yerr_mcintosh = np.array([[29, 21]]).T
    plt.errorbar(datetime(2025,1,1) + timedelta(days=213.), 233, yerr=asymmetric_yerr_mcintosh, label='McIntosh, et al. 2020', fmt='co', capsize=5) # https://link.springer.com/article/10.1007/s11207-020-01723-y?utm_medium=affiliate&CJEVENT=12d7ea2df8b311ec8243005c0a82b824&utm_campaign=CONR_BOOKS_ECOM_GL_PHSS_ALWYS_DEEPLINK&utm_content=textlink&utm_source=commission_junction&utm_term=PID100052171
    # Axes/labels/saving:
    plt.xlabel('Date')
    plt.ylabel('$S_{\mathrm{N}}$')
    plt.title('Solar Cycle Hindcast and Forecast', fontsize=18)
    plt.legend(loc='best', framealpha=1)
    plt.savefig(figures_directory + 'solarCycleHindcastForecast.png', dpi=300)

    sys.exit(0)