# This code contains all that is needed to automatically run FOCI to fit different models to all possible combinations
# of a series of model potential model drivers and a single response variable.

# Top-level Imports:
import numpy as np
from multiprocessing import shared_memory
from functools import partial
import time
from multiprocessing import cpu_count
from itertools import repeat
from multiprocessing import Pool
from sklearn.neighbors import NearestNeighbors
import scipy.stats as ss
import itertools
from functools import reduce
from sklearn.metrics import mean_squared_error
from pygam import LinearGAM, l, f, s, te
from tqdm import tqdm
from math import sqrt
from sklearn.model_selection import KFold
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')

# Local imports:
from tools import solarToolbox

# Functions:
def relate(drivers, true_data, subset_number, test_data, lambda_range=None):
    '''
    This is the master routine for implementing FOCI. It does the following:
        1. Takes in an array where each column is a predictor variable and computes all possible cross-terms from those
        predictor variables.
        2. Runs FOCI (Azadkia and Chatterjee, 2021) to pre-screen the predictor variables down to a chosen subset
        defined by the user.
        3. Fits GAMs (w/ extensive Cross-Validation) between the downselected predictor variables and the response variable.
        4. The best GAM is chosen based on Information Theoretic metrics.
        5. Returns the best drivers, the best GAM, and statistics on the model performance.
    :param drivers: list
        A data list where each row is an observation and each column is a different predictor variable.
        Note that each row has three elements: element[0] is a string describing the variable, element[1] is an array or
        list of variable times, and element[2] is the actual variable.
    :param true_data: list
        A list for the response variable to be predicted or correlated to the predictor variables. Has the same format
        as a single element in 'drivers'.
    :param subset_number: int or arraylike
        The number of features with which to downselect the predictor variables. Must be equal to or less than the
        number of predictor variables themselves. If a list is given, it MUST have two elements, and the FOCI will be
        run twice. It will be run once to downselect to number of variables corresponding to the first element, and then
        it will be run again with all possible cross-terms of the initial subset.
    :param test_data: list or arraylike
        Data to feed to the final model for prediction.
    :param lambda_range: list
        A list with two elements that specifies the range of lambda values to loop over when performing regularization
        on the GAMs. Default is None, in which case the range will be [0, 1e6].
    :return best_drivers:
        The best model drivers.
    :return best_gam:
        The best Generalized Additive Model.
    :return best_statistics:
        Statistics on how the best model performs.
    '''
    response = true_data[-1]

    # Step 1 - Compute all possible cross-terms to get the full span of predictor variables:
    combined_drivers = drivers
    for i in range(len(combined_drivers)):
        combined_drivers[i][-2] = list(combined_drivers[i][-2]) + [combined_drivers[i][-2][-1] + 1]
        combined_drivers[i][-1] = list(combined_drivers[i][-1]) + [test_data[i]]
    all_drivers_initial = get_cross_terms(combined_drivers) # This includes the testing item as well...

    # An extra (but likely necessary) step - scale all the inputs to the same order of magnitude as the response variable:
    reference_order_of_magnitude = solarToolbox.orderOfMagnitude(np.nanmean(true_data[-1]))
    all_drivers_modified = all_drivers_initial
    for item in all_drivers_modified:
        current_order_of_magnitude = solarToolbox.orderOfMagnitude(np.abs(np.nanmean(item[-1])))
        needed_change = reference_order_of_magnitude - current_order_of_magnitude
        item[-1] = [element * 10 ** (needed_change) for element in item[-1]]
    # Extracting only the testing item...
    all_drivers_test = []
    for item in all_drivers_modified:
        all_drivers_test.append([item[0], item[1][-1], item[2][-1]])
    # Exclude the testing item now...
    all_drivers = []
    for item in all_drivers_modified:
        all_drivers.append([item[0], item[1][:-1], item[2][:-1]])

    # Step 2 - Run FOCI on the drivers. Do so in two steps if specified by the user:
    if type(subset_number) == int:
        indices = foci_main_38(response, np.asarray([element[-1] for element in all_drivers]).T, num_features=subset_number)
        downselected_input_data = []
        downselected_test_data = []
        for idx in indices:
            downselected_input_data.append(all_drivers[idx])
            downselected_test_data.append(all_drivers_test[idx])
    else:
        if len(subset_number) == 2:
            print('Running FOCI in two steps...')
            time.sleep(1)
            indices_1 = foci_main_38(response, np.asarray([element[-1] for element in all_drivers]).T, num_features=subset_number[0])
            first_subset = [all_drivers[element] for element in indices_1]
            first_subset_test = [all_drivers_test[element] for element in indices_1]
            all_subsetted_drivers = get_cross_terms(first_subset)
            indices_2 = foci_main_38(response, np.asarray([element[-1] for element in all_subsetted_drivers]).T, num_features=subset_number[1])
            downselected_input_data = []
            downselected_test_data = []
            for idx in indices_2:
                downselected_input_data.append(all_subsetted_drivers[idx])
                downselected_test_data.append(first_subset_test[idx])
        else:
            raise ValueError('You can only supply at most TWO subset numbers.')

    # Step 3: Fit GAMs, with extensive cross-validation:
    # Naively...
    X = np.array([element[-1] for element in downselected_input_data]).T
    y = true_data[-1]
    gam = LinearGAM(s(0) + s(1) + s(2)).fit(X, y)
    res24 = gam.predict(np.array([X[-1, :]])) # Naive SC24 'prediction'
    res25 = gam.predict(np.array([[element[-1] for element in downselected_test_data]])) # Naive SC25 prediction (amplitude)

    # Rigorously:
    print('Tuning the smoothing parameter...')
    numLambdas = 1000
    if lambda_range is None:
        lambdas = np.linspace(0, 1e6, numLambdas)
    else:
        lambdas = np.linspace(lambda_range[0], lambda_range[-1], numLambdas) # np.linspace(0.014432,0.014434, numLambdas) # Time # np.linspace(0.01328, 0.01329, numLambdas) # Amplitude
    numTerms = X.shape[1]
    numKnobs = X.shape[0] * 2
    arguments = modelArgs(numTerms, model='s', knobs=numKnobs)
    models = []
    model_error = []
    aicc_vals = []
    for lam in tqdm(lambdas):
        gam = LinearGAM(eval(arguments), lam=lam).fit(X, y)
        models.append(gam)
        res = gam.predict(X)
        mse = mean_squared_error(true_data[-1], res)
        model_error.append(mse)
        my_aicc = gam._estimate_AICc(mu=res, y=true_data[-1])
        aicc_vals.append(my_aicc)
    best_model_ind = np.argmin(aicc_vals)
    myGam = models[best_model_ind]
    print('Best model found; best lambda = '+str(lambdas[best_model_ind]))
    # View the model results:
    # plt.figure()
    # plt.plot(lambdas, aicc_vals)
    # plt.xlabel('Lambda')
    # plt.ylabel('AICc')

    # # More Rigourously...
    # print('Fitting GAMs...')
    # time.sleep(1)
    # # TODO: Fix the fitting - the results obtained in lines 127 and 128 are close to zero, and smack of an incorrect fitting procedure!
    # bestModelInfo, bestModelDrivers = modelLoop(truth=true_data, input=downselected_input_data,
    #                                                                 modelType='GAM', sparse=True)

    # Predictions...
    res24_final = myGam.predict([X[-1, :]])
    res25_final = myGam.predict(np.array([[element[-1] for element in downselected_test_data]]))

    # Uncertainty...
    res24_CI = myGam.prediction_intervals([X[-1, :]], width=0.68)
    res25_CI = myGam.prediction_intervals(np.array([[element[-1] for element in downselected_test_data]]), width=0.68)

    # Print results:
    print('SC24 hindcast: '+str(res24_final[0])+' 68% CI: '+str(res24_CI[0]))
    print('SC25 forecast: ' + str(res25_final[0]) + ' 68% CI: ' + str(res25_CI[0]))

    return myGam, downselected_input_data, [res24_final[0], res25_final[0]], [res24_CI[0], res25_CI[0]]

def get_cross_terms(data):
    '''
    Given some 2D array (rows being observations and columns being individual data streams), compute all possible cross-
    terms and return them, along with the original data streams.
    :param data: numpy.ndarray
        The data with which to obtain cross-terms.
    :return expanded_data: numpy.ndarray
        The data with all possible cross-terms included.
    '''
    print('Computing cross-terms...')
    all_cross_terms_indices = uniqueCombs(data)
    all_cross_terms = []  # Each element itself has two elements: (0) time values, (1) cross-term value
    j = 0
    for indices in all_cross_terms_indices:
        print('Iteration: ' + str(j + 1) + '/' + str(len(all_cross_terms_indices)))
        currentTerms = [data[element] for element in indices]
        currentTermsNames = [element[0] for element in currentTerms]
        startTimes = [min(element[1]) for element in currentTerms]
        endTimes = [max(element[1]) for element in currentTerms]
        boundaries = [max(startTimes), min(endTimes)]
        time_template = np.asarray(currentTerms[0][1])[np.where(
            (np.asarray(currentTerms[0][1]) >= boundaries[0]) & (np.asarray(currentTerms[0][1]) <= boundaries[-1]))[0]]
        currentTerms_subset = []
        for i in range(len(currentTerms)):
            current_terms_subset_time, current_terms_subset_vals = solarToolbox.subset_data(currentTerms[i][1],
                                                                                  currentTerms[i][2],
                                                                                  boundaries=boundaries)
            currentTerms_subset.append([currentTerms[i][0], current_terms_subset_time, current_terms_subset_vals])
        # Force temporal uniformity using np.intersect1d:
        subTimes = [element[1] for element in currentTerms_subset]
        subVals = [element[2] for element in currentTerms_subset]
        subInds = [np.linspace(0, len(element) - 1, len(element)) for element in subTimes]
        subIndsTimes = [(x, y) for x, y in zip(subInds, subTimes)]
        myTup = tuple([element[1] for element in currentTerms_subset])
        result = reduce(np.intersect1d, myTup)
        # Indices of valid values for all data streams:
        validInds = [np.intersect1d(result, element, return_indices=True)[2] for element in subTimes]
        cleanSubData = [np.asarray(x)[y] for x, y in zip(subVals, validInds)]
        cross_term = np.prod(np.asarray(cleanSubData).T, axis=1)  # np.prod(terms_subset_data_only, axis=1)
        all_cross_terms.append(['_'.join(currentTermsNames), result,
                                cross_term])  # The first element tells which drivers make up the cross-term, the second contains the timestamps, and the third the cross-term values.
        j += 1
    print('Cross-terms obtained.')
    # Combine the cross-terms with the original input data into a single data structure:
    expanded_data = data + all_cross_terms
    return expanded_data

def uniqueCombs(arraylike, exclude_singular=True):
    """
    Given some list or array, return all the unique combinations of the elements of the input (in terms of indices).
    :param arraylike: list or ndarray
        Some univariate data.
    :param exclude_singular: bool
        If True, returns only those combinations that DO NOT include the zero set and sets with only one element.
        Default is True.
    :return uniqCombs:
        A list of the unique combinations.
    """
    index_list = np.array([int(num) for num in np.linspace(0, len(arraylike) - 1, num=len(arraylike))])
    # Generate unique combinations of those inputs (order is irrelevant):
    myComb = []
    for L in range(len(index_list) + 1):
        for subset in itertools.combinations(index_list, L):
            myComb.append(subset)
    if exclude_singular:
        # Ignore combinations that consist of the zero set and sets with only one element:
        uniqCombs = [list(element) for element in myComb if len(element) > 1]
        return uniqCombs
    else:
        return myComb

def modelLoop(truth, input, modelType='GLM', sparse=False):
    """
    Given some input data, fit a series of models between the input data and the truth data that is the desired
    output of the model. From the input data, return the linear model that BEST models the data. Perform
    cross-validation for model selection using the 90-10-10 rule.
    :param truth: list
        A two-element list where the first element is a string describing the data,
        the second element is an array-like of datetime values for the data, and the second
        element are data values, with the same length as the array-like of time values.
    :param input: list
        A multiple-element list where each element has the same format as 'truth', but corresponds to a different
        input variable.
    :param modelType: str
        A string describing the type of model to fit. Valid arguments include 'GLM' for Generalized Linear Model and
        'GAM' for Generalized Additive Model. Default is 'GLM'.
    :param sparse: bool
        Controls whether standard 90-10-10 K-fold cross-validation is used during model fitting. Will do so if set to
        False. If True, simply does K-fold cross-validation with the 'leave one out approach'. Default is False.
    :return modelResults: list
        Contains details about the best model. The elements of the list are as follows:
        0: Index of the 90-10-10 split chosen as best-performing
        1: LinearGAM (GLM) object
        2: Absolute Percentage Error values
        3: Mean Absolute Percentage Error
        4: Bias values (residuals)
        5: Root Mean Square Error values
        6: Pearsons' Correlation Coefficient
        7: Mean Squared Error
        8: List of model drivers
    """
    unique_combinations = uniqueCombs(input)
    modelResults = []
    print('Fitting '+modelType+'s with ' + str(len(unique_combinations)) + ' combinations of drivers...')
    for i in tqdm(range(len(unique_combinations))):
        currentDrivers = []
        for j in unique_combinations[i]:
            currentDrivers.append(input[j])
        if sparse == False:
            res, coverage = fitModel(truth, currentDrivers, modelType=modelType)
        else:
            res, coverage = fitSparseModel(truth, currentDrivers, modelType=modelType)
        modelResults.append( [res, coverage] )

    # Select the best model:
    modelResults_res = [element[0] for element in modelResults]
    # modelResults_cov = [element[1] for element in modelResults]
    # errorMetrics = [element[1] for element in modelResults_res]
    # MSE_vals = [element[-1] for element in errorMetrics]
    AICc_vals = [element[-1] for element in modelResults_res]
    bestModelInd = np.argmin(AICc_vals)
    indices_of_best_drivers = unique_combinations[bestModelInd]
    driver_strings = [element[0] for element in input]
    best_drivers = []
    for idx in indices_of_best_drivers:
        best_drivers.append(driver_strings[idx])
    print('Fitting complete. Best '+modelType+' identified.\n')

    return modelResults[bestModelInd], best_drivers

def fitSparseModel(truth, my_drivers, modelType='GAM'):
    """
    Fit a linear or GAM model with PyGAM, for sparse data.
    :param truth: list
        A two-element list where the first element is a string describing the data,
        the second element is an arraylike of time values for the data, and the third
        element are data values, with the same length as the arraylike of datetimes.
    :param my_drivers: list
        A multiple-element list where each element has the same format as 'truth', but corresponds to a different
        input variable.
    :param modelType: str
        A string describing the type of model to fit. Valid arguments include 'GLM' for Generalized Linear Model and
        'GAM' for Generalized Additive Model. Default is 'GLM'.
    :return final_cross_val_results: list
        Contains the model object and performance statistics for the model
    :return coverage: list
        A list where the first element are datetimes corresponding to the places of valid data across the entire fitting
        interval, the second are the valid driver data, and the last are the valid true data.
    """
    drivers = np.array([element[-1] for element in my_drivers]).T
    numLambdas = 100
    lambdas = np.linspace(0, 1000000, numLambdas)
    numTerms = len(my_drivers)
    cross_val_results = []
    scores = []
    obj = 'AICc'
    if modelType == 'GLM':
        modStr = 'l'  # Linear terms
        arguments = modelArgs(numTerms, model=modStr)
    elif modelType == 'GAM':
        modStr = 's'  # Spline terms
        arguments = modelArgs(numTerms, model=modStr, knobs=1000)  # knobs=len(train_index)
    else:
        raise ValueError('Invalid value given for argument "modelType". Argument must either be "GLM" or "GAM".')

    for k, lam in tqdm(enumerate(lambdas)):
        # See line 2037 of https://github.com/dswah/pyGAM/blob/master/pygam/pygam.py
        # TODO: A THIRD loop could be placed here, around the GAM model fit, to implement an l1 penalty to account for non-Gaussian behavior of residuals...
        model = LinearGAM(eval(arguments), lam=lam).fit(drivers, truth[1])
        # lams = [lam] * numTerms
        # model.gridsearch(drivers_subset_data_only[train_index, :], truth_subset_data_only[train_index], lam=lams)
        # Pick an objective obj to optimize via cross-validation
        # E.g., ['GCV', 'UBRE', 'AIC', 'AICc']
        # AICc is preferred (use internal method to calculate it)
        # score = model.statistics_[obj]
        # scores[j, k] = score
        preds = model.predict(drivers)
        actual = truth[1]
        my_aicc = model._estimate_AICc(mu=preds, y=actual)
        scores.append(my_aicc) # mean_squared_error(actual, preds) # my_aicc
        modelStats = errorStats(preds, actual, printResults=False)
        modelPerformance = [*modelStats, mean_squared_error(actual, preds)]
        cross_val_results.append([model, modelPerformance, my_aicc])

    # Perform one more fit, but using ALL the data, but with the value of lambda corresponding to the best model:
    mean_scores = np.mean(scores, axis=0) # We take the mean across the folds in order to maximize generalizability.
    best_single_score_ind = np.argmin(mean_scores)
    best_lambda = lambdas[best_single_score_ind]
    final_model = LinearGAM(eval(arguments), lam=best_lambda).fit(drivers, truth[1])

    # Evaluate the performance of this model across the entire dataset:
    finalPreds = final_model.predict(drivers)
    final_modelStats = errorStats(finalPreds, truth[1])
    final_model_performance = [*final_modelStats, mean_squared_error(truth[1], finalPreds)]
    final_cross_val_results = [final_model, final_model_performance, final_model.statistics_[obj]]

    # # Select the best-performing model (according to MEAN SQUARED ERROR):
    # modelStats = [element[-1] for element in cross_val_results]
    # MSE_vals = [element[-1] for element in modelStats]
    # bestModelInd = np.argmin(MSE_vals)

    coverage = [truth[1], drivers, truth[2], finalPreds]

    return final_cross_val_results, coverage

def fitModel(truth, drivers, modelType='GLM'):
    """
    Fit a linear or GAM model with PyGAM, given some input data and truth data to fit to.
    :param truth: list
        A two-element list where the first element is a string describing the data,
        the second element is an arraylike of datetime values for the data, and the third
        element are data values, with the same length as the arraylike of datetimes.
    :param drivers: list
        A multiple-element list where each element has the same format as 'truth', but corresponds to a different
        input variable.
    :param modelType: str
        A string describing the type of model to fit. Valid arguments include 'GLM' for Generalized Linear Model and
        'GAM' for Generalized Additive Model. Default is 'GLM'.
    :return final_cross_val_results: list
        Contains the model object and performance statistics for the model
    :return coverage: list
        A list where the first element are datetimes corresponding to the places of valid data across the entire fitting
        interval, the second are the valid driver data, and the last are the valid true data.
    """
    # Harmonize the data in time before doing any fitting:
    # 1: Isolate the data between intersecting starting and ending dates:
    startTimes = [*[min(truth[1])], *[min(element[1]) for element in drivers]]
    endTimes = [*[max(truth[1])], *[max(element[1]) for element in drivers]]
    boundaries = [max(startTimes), min(endTimes)]
    truth_subset_time, truth_subset_vals = solarToolbox.subset_data(truth[1], truth[2], boundaries=boundaries)
    truth_subset = [truth[0], truth_subset_time, truth_subset_vals]
    drivers_subset = []
    for i in range(len(drivers)):
        current_driver_subset_time, current_driver_subset_vals = solarToolbox.subset_data(drivers[i][1], drivers[i][2],
                                                                             boundaries=boundaries)
        drivers_subset.append([drivers[i][0], current_driver_subset_time, current_driver_subset_vals])
    # 2: Find the common indices corresponding to the intersection of ALL the subset data:
    subTimes = [element[1] for element in drivers_subset]
    subVals = [element[2] for element in drivers_subset]
    # subInds = [np.linspace(0, len(element) - 1, len(element)) for element in subTimes]
    # subIndsTimes = [(x, y) for x, y in zip(subInds, subTimes)]
    myTup = tuple([element[1] for element in drivers_subset])
    result = reduce(np.intersect1d, myTup)
    # Indices of valid values for all data streams:
    validInds = [np.intersect1d(result, element, return_indices=True)[2] for element in subTimes]
    cleanSubData = [np.asarray(x)[y] for x, y in zip(subVals, validInds)]
    drivers_subset_data_only = np.asarray(cleanSubData).T # np.asarray([element[2] for element in drivers_subset]).T
    truthInds = np.intersect1d(truth_subset[1], result, return_indices=True)[1]
    truth_subset_data_only = np.asarray(truth_subset[-1][truthInds]) # np.asarray(truth_subset[-1])

    # Perform model fitting and perform cross-validation with the 90-10-10 rule:
    numTerms = drivers_subset_data_only.shape[1]
    # if modelType == 'GLM':
    #     modStr = 'l' # Linear terms
    #     arguments = modelArgs(numTerms, model=modStr)
    # elif modelType == 'GAM':
    #     modStr = 's' # Spline terms
    #     arguments = modelArgs(numTerms, model=modStr, knobs=500)
    # else:
    #     raise ValueError('Invalid value given for argument "modelType". Argument must either be "GLM" or "GAM".')
    # arguments = modelArgs(numTerms, model=modStr)
    kf = KFold(n_splits=10, shuffle=True)
    kf.get_n_splits(drivers_subset_data_only)
    cross_val_results = []
    obj = 'AICc'
    numLambdas = 20
    # lambdas = np.arange(0, 105000, 5000) # np.linspace(0, 1, num=11, endpoint=True) # Consider a MUCH wider search space, i.e.: np.arange(0, 105000, 5000)
    lambdas = np.linspace(0, 1000000, numLambdas)
    scores = np.zeros((10, lambdas.shape[0]), dtype=float)
    for j, (train_index, test_index) in enumerate(kf.split(drivers_subset_data_only)):
        # print(f"Fold {j+1}:")
        # print(f"  Train: index={train_index}")
        # print(f"  Test:  index={test_index}")
        # lambdas = np.linspace(0, 1, num=1, endpoint=True)
        if modelType == 'GLM':
            modStr = 'l'  # Linear terms
            arguments = modelArgs(numTerms, model=modStr)
        elif modelType == 'GAM':
            modStr = 's'  # Spline terms
            arguments = modelArgs(numTerms, model=modStr, knobs=1000) # knobs=len(train_index)
        else:
            raise ValueError('Invalid value given for argument "modelType". Argument must either be "GLM" or "GAM".')
        # One way to choose lambda is with manual cross validation loops
        # Another way is to use a grid search via AICc or GCV
        # obj = 'AICc'
        # lambdas = np.linspace(0.6, 0.6, num=1)
        for k, lam in enumerate(lambdas):
            # See line 2037 of https://github.com/dswah/pyGAM/blob/master/pygam/pygam.py
            # TODO: A THIRD loop could be placed here, around the GAM model fit, to implement an l1 penalty to account for non-Gaussian behavior of residuals...
            model = LinearGAM(eval(arguments), lam=lam).fit(drivers_subset_data_only[train_index, :], truth_subset_data_only[train_index])
            # lams = [lam] * numTerms
            # model.gridsearch(drivers_subset_data_only[train_index, :], truth_subset_data_only[train_index], lam=lams)
            # Pick an objective obj to optimize via cross-validation
            # E.g., ['GCV', 'UBRE', 'AIC', 'AICc']
            # AICc is preferred (use internal method to calculate it)
            # score = model.statistics_[obj]
            # scores[j, k] = score
            preds = model.predict(drivers_subset_data_only[test_index, :])
            actual = truth_subset_data_only[test_index]
            my_aicc = model._estimate_AICc(mu=preds, y=actual)
            scores[j, k] = my_aicc # mean_squared_error(actual, preds) # my_aicc
            modelStats = errorStats(preds, actual, printResults=False)
            modelPerformance = [*modelStats, mean_squared_error(actual, preds)]
            cross_val_results.append([j, model, modelPerformance, my_aicc])

    # View the scores (as a sanity check)
    # fig, axs = plt.subplots(1, 1)
    # num_folds = scores.shape[0]
    # for ii in range(num_folds):
    #     aic_vals = scores[ii]
    #     axs.plot(lambdas, aic_vals - np.mean(aic_vals), marker='o')#, aspect='auto')
    # plt.xlabel('Lambda')
    # plt.ylabel('Fold')
    # plt.colorbar()

    # Perform one more fit, but using ALL the data, but with the value of lambda corresponding to the best model:
    mean_scores = np.mean(scores, axis=0) # We take the mean across the folds in order to maximize generalizability.
    best_single_score_ind = np.argmin(mean_scores)
    best_lambda = lambdas[best_single_score_ind]
    final_model = LinearGAM(eval(arguments), lam=best_lambda).fit(drivers_subset_data_only, truth_subset_data_only)

    # if drivers_subset_data_only.shape[1] == 2:
    #     first_feature = drivers_subset_data_only[:, 0]
    #     second_feature = drivers_subset_data_only[:, 1]
    #     xs = np.linspace(np.min(first_feature), np.max(first_feature), 300)
    #     ys = np.linspace(np.min(second_feature), np.max(second_feature), 500)
    #
    #     XX, YY = np.meshgrid(xs, ys)
    #     grid_data = np.vstack([XX.ravel(), YY.ravel()]).T
    #
    #     z_flat = final_model.predict(grid_data)
    #     ZZ = z_flat.reshape(XX.shape)
    #
    #     fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    #     ax.plot_surface(XX, YY, ZZ)
    #     ax.scatter(first_feature, second_feature, truth_subset_data_only, marker='o')
    #
    #     fhat = final_model.predict(drivers_subset_data_only)
    #     res = fhat - truth_subset_data_only
    #
    #     plt.figure()
    #     plt.hist(res, bins = 50)
    #     pass

    # Evaluate the performance of this model across the entire dataset:
    finalPreds = final_model.predict(drivers_subset_data_only)
    final_modelStats = errorStats(finalPreds, truth_subset_data_only)
    final_model_performance = [*final_modelStats, mean_squared_error(truth_subset_data_only, finalPreds)]
    final_cross_val_results = [final_model, final_model_performance, final_model.statistics_[obj]]

    # # Select the best-performing model (according to MEAN SQUARED ERROR):
    # modelStats = [element[-1] for element in cross_val_results]
    # MSE_vals = [element[-1] for element in modelStats]
    # bestModelInd = np.argmin(MSE_vals)

    coverage = [result, drivers_subset_data_only, truth_subset_data_only, finalPreds]

    return final_cross_val_results, coverage

def errorStats(y_pred, y_true, printResults=True):
    """
    Compute various error statistics between two univariate time series.
    :param y_pred: arraylike
        Estimated values.
    :param y_true: arraylike
        True values
    :param printResults: Bool
        If True, prints all the error metrics. Default is True.
    :return apes:
        The Absolute Percentage error for the predictions (where valid).
    :return mapeVal:
        The Mean Absolute Percentage Error for the predictions.
    :return bias:
        The average value of the residual for the predictions.
    :return rmse:
        The RMSE for the predictions.
    :return corr:
        The correlation between the predictions and the truth.
    """
    # Compute error statistics:
    apes = ape(y_pred, y_true)
    mapeVal = mape(y_pred, y_true)
    bias = np.subtract(y_pred, y_true)
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    corr = np.corrcoef(y_pred, y_true)[0, 1]
    if np.isnan(corr) == True:
        corr = ma.corrcoef(ma.masked_invalid(y_pred), ma.masked_invalid(y_true))[0, 1]

    # Print results:
    if printResults:
        print('MAPE: ' + str(mapeVal) + '%')
        print('Mean Bias: ' + str(np.nanmean(bias)))
        print('RMSE: ' + str(rmse))
        print('Correlation: ' + str(corr) + '\n')

    return apes, mapeVal, bias, rmse, corr

def ape(obs, true):
    """
    Compute the percentage error between two arrays.
    :param x: arraylike
        The modeled values.
    :param true: arraylike
        The actual (true) values.
    :return ape: arraylike
        The absolute percentage error between each value
    """
    ape = np.zeros_like(obs)
    for i in range(len(obs)):
        ape[i] = np.abs( (obs[i] - true[i]) / true[i])*100
    return ape

def mape(obs, true):
    """
    Compute the mean absolute percentage error between two arrays.
    :param obs: arraylike
        The modeled or observed values.
    :param true: arraylike
        The actual (true) values.
    :return mape: float
        The mean absolute percentage error.
    """
    n = len(obs)
    quantity = np.abs(np.divide(np.subtract(np.asarray(true), np.asarray(obs)), np.asarray(true)))
    mape = (1./n) * np.sum(quantity) * 100
    return mape

def modelArgs(numTerms, model='s', knobs=20):
    """
    Symbolically return the terms for a model to be fit by pyGAM
    :param numTerms: int
        The number of terms to include.
    :param model: str
        The type of model to build terms out of. Valid arguments include 's', 'l', 'f', and 'te'.
    :param knobs: int
        The number of basis functions to use. Default is 20.
    :return args: str
        The model string.
    """
    args = model+'(0)'
    for i in range(numTerms-1):
        if knobs == 20:
            args += '+'+model+'('+str(i+1)+')'
        else:
            if model == 's' or model == 'te':
                args += '+'+model+'('+str(i+1)+', n_splines='+str(knobs)+')'
            else:
                args += '+' + model + '(' + str(i + 1) + ')'
    return args

#-----------------------------------------------------------------------------------------------------------------------
# FOCI Code (Erick F. Vega, Brian J. Thelen, and Joel W. LeBlanc):
# TODO: All of the code below needs to be commented comprehensively.
def shared_mem_helper(subset_inds, Y_name, X_name, Y_shape, X_shape, X_dtype, Y_dtype):
    X_sh_mem = shared_memory.SharedMemory(name=X_name)
    Y_sh_mem = shared_memory.SharedMemory(name=Y_name)

    X = np.ndarray(X_shape, dtype=X_dtype, buffer=X_sh_mem.buf)
    Y = np.ndarray(Y_shape, dtype=Y_dtype, buffer=Y_sh_mem.buf)

    Q = _estimateQ(Y, X[:, subset_inds])

    return Q

def foci_main_38(Y, X, num_features = None, stop = True):
    n = len(Y)
    p = X.shape[1]
    Q = np.zeros(num_features)
    index_select = np.zeros(num_features, dtype=int)

    # Generates the things we'll need for sharing memory
    X_shm_buffer = shared_memory.SharedMemory(create=True, size=X.nbytes)
    Y_shm_buffer = shared_memory.SharedMemory(create=True, size=Y.nbytes)
    X_shape = X.shape
    Y_shape = Y.shape
    X_dtype = X.dtype
    Y_dtype = Y.dtype
    X_name = X_shm_buffer.name
    Y_name = Y_shm_buffer.name

    X_shared_memory = np.ndarray(X_shape, dtype=X_dtype, buffer=X_shm_buffer.buf)
    Y_shared_memory = np.ndarray(Y_shape, dtype=Y_dtype, buffer=Y_shm_buffer.buf)

    X_shared_memory[:] = X[:]
    Y_shared_memory[:] = Y[:]

    #
    memory_args = [Y_name, X_name, Y_shape, X_shape, X_dtype, Y_dtype]
    subset_inds = range(num_features)
    args = zip(repeat(memory_args), subset_inds)

    parallel_start = time.time()
    with Pool(processes=cpu_count() - 1) as pool:
        seq_Q = pool.map(partial(shared_mem_helper, Y_name = Y_name, X_name = X_name, Y_shape = Y_shape, X_shape = X_shape, X_dtype = X_dtype, Y_dtype = Y_dtype), range(num_features))
        # out = pool.starmap(shared_mem_helper, args)
    parallel_stop = time.time()
    parallel_ellapsed = parallel_stop-parallel_start


    # shared_mem_helper(Y_name, X_name, Y_shape, X_shape, X_dtype, Y_dtype)
    # seq_Q = np.empty(num_features)
    #
    # serial_start = time.time()
    # for q_ind in range(num_features):
    #     seq_Q[q_ind] = _estimateQ(Y, X[:, q_ind])
    # serial_stop = time.time()
    # serial_ellapsed = serial_stop-serial_start

    Q[0] = np.max(seq_Q)
    index_max = np.min(np.where(seq_Q == Q[0]))
    index_select[0] = index_max

    count = 1

    with Pool(processes=cpu_count() - 1) as pool:
        while count < num_features:
            # print(count)
            seq_Q = np.zeros(p - count)
            index_left = np.setdiff1d(np.arange(p), index_select[0:count])

            subset_inds = [np.append(index_select[:count], indx) for indx in index_left]
            parallel_start = time.time()
            seq_Q = pool.map(partial(shared_mem_helper, Y_name=Y_name, X_name=X_name, Y_shape=Y_shape, X_shape=X_shape,
                                   X_dtype=X_dtype, Y_dtype=Y_dtype), subset_inds)
            parallel_stop = time.time()
            parallel_ellapsed = parallel_stop-parallel_start
            # print(parallel_ellapsed)
            # def estimateQFixedYandSubX(indx):
            #     selected_indices = np.append(index_select[:count], indx)
            #     return _estimateQ(Y, X[:, selected_indices])

            # seq_Q = pool.map(estimateQFixedYandSubX, index_left)

            Q[count] = max(seq_Q)
            index_max = np.min(np.where(seq_Q == Q[count]))
            index_select[count] = index_left[index_max]
            count = count + 1

    X_shm_buffer.close()
    Y_shm_buffer.close()

    X_shm_buffer.unlink()
    Y_shm_buffer.unlink()
    return index_select

def _estimateQ(Y, X):
    # For now, we'll only handle the case without any repeats
    X.reshape(-1, 1)
    n = len(Y)
    if len(X.shape) == 1:
        nn_X = NearestNeighbors(n_neighbors=3).fit(X.reshape(-1, 1))
        nn_dists, nn_idx = nn_X.kneighbors(X.reshape(-1, 1))
    else:
        nn_X = NearestNeighbors(n_neighbors=3).fit(X)
        nn_dists, nn_idx = nn_X.kneighbors(X)

    nn_index_X = nn_idx[:, 1]

    R_Y = ss.rankdata(Y, method = "max")
    L_Y = ss.rankdata(-Y, method = "max")

    Q_n = np.sum(np.minimum(R_Y, R_Y[nn_index_X]) - (L_Y**2)/n) / (n**2)

    return Q_n