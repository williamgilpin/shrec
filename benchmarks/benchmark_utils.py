import numpy as np
import scipy
import warnings

from shrec.utils import *


import pandas as pd
import warnings
import numpy as np

from scipy.signal import coherence
from scipy.stats import spearmanr, pearsonr, kendalltau

from darts import TimeSeries
import darts.metrics.metrics

import dtw

from mutual_info import mutual_information, conditional_information
from phase_synchrony import sync_average, find_instaneous_sync

def print_dict(a):
    for row in a:
        print(np.round(a[row], 2), row, flush=True)
    print("\n")
    
    
    
import numpy as np


def fft_phase(a):
    ns = len(a)
    af = np.fft.fft(a)
    af = af / np.abs(af)
    af = af[:ns // 2]
    angs = np.arctan2(np.imag(af), np.real(af))
    return angs
     
def coherence_phase(a, b, freq=1, FS=1):
    ap, bp = fft_phase(a), fft_phase(b)
    return np.sqrt((np.cos(ap) + np.cos(bp))**2 + (np.sin(ap) + np.sin(bp))**2) / 2


def score_ts(true_y, pred_y):
    """
    Score a pair of time series
    """
    true_yn, pred_yn = (
        np.squeeze(standardize_ts(true_y)), 
        np.squeeze(standardize_ts(pred_y))
    )
    metric_list = [
        #'coefficient_of_variation',
        'mae',
        'mape',
        'marre',
        'mse',
        'r2_score',
        'rmse',
        'smape'
    ]

    scores = dict()
    
    kval = 30
    np.random.seed(0)
    lo, hi = (
        mutual_information((np.random.permutation(true_yn)[:, None],
                            np.random.permutation(true_yn)[:, None]),
                           k=kval
                          ), 
        mutual_information((true_yn[:, None], 
                            true_yn[:, None]), 
                           k=kval), 
    )
    mi = mutual_information((true_yn[:, None], pred_yn[:, None]), k=kval)
    scores["mutual_info"] = (mi - lo) / (hi - lo)
    scores["conditional_info"] = conditional_information(true_y, pred_y, k=kval)
    scores["conditional_info_back"] = conditional_information(pred_y, true_y, k=kval)
    
    
    
    true_y_df = TimeSeries.from_dataframe(pd.DataFrame(np.squeeze(true_y)))
    pred_y_df = TimeSeries.from_dataframe(pd.DataFrame(np.squeeze(pred_y)))
    pred_y_df_neg = TimeSeries.from_dataframe(pd.DataFrame(np.squeeze(-pred_y)))
    for metric_name in metric_list:
        metric_func = getattr(darts.metrics.metrics, metric_name)
        try:
            if metric_name in ['r2_score']:
                scores[metric_name] = max(metric_func(true_y_df, pred_y_df), metric_func(true_y_df, pred_y_df_neg))
            else:
                scores[metric_name] = min(metric_func(true_y_df, pred_y_df), metric_func(true_y_df, pred_y_df_neg))
        except:
            print(metric_name, " Skipped")
    
    corr = spearmanr(true_y, pred_y)
    scores["spearman"] = corr.correlation
    corr = pearsonr(true_y, pred_y)[0]
    scores["pearson"] = corr
    corr = kendalltau(true_y, pred_y)[0]
    scores["kendalltau"] = corr
    
    
    scores["sync"] = max(sync_average(true_yn, pred_yn), sync_average(true_yn, -pred_yn))
    scores["coherence"] = np.mean(coherence(true_y, pred_y)[1])
    scores["coherence_phase"] = np.mean(coherence_phase(true_y, pred_y)[1])
    
    scores["cross forecast error"] = cross_forecast(pred_y, true_y, split=0.75)
    #scores["cross forecast error rf"] = cross_forecast(pred_y, true_y, model="rf", split=0.75)
    scores["cross forecast error neural"] = cross_forecast(pred_y, true_y, model="mlp", split=0.75)
    
    
    scores["cross forecast error neural 2"] = cross_forecast(pred_y, true_y, model="mlp")

    scores["dynamic time warping distance"] = min(dtw.dtw(pred_y, true_y).normalizedDistance, dtw.dtw(-pred_y, true_y).normalizedDistance)
    
    
    return scores

from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

def cross_forecast(ts_input, ts_target, tau=50,
                   model="ridge", 
                   split=None,
                   return_points=False,
                   return_model=False):
    """
    Train a forecast model that predicts a target time series using timepoints
    of a given time series. 
    Currently, this function should only be used for linear
    models (such as for Granger causality) because it does not split train and test
    
    split (float): the fraction of the data to use as a test split
    
    Development
        add a train/test forecast split in order to faciliate training 
        nonlinear models
    """
    sort_inds = np.argsort(ts_input)
    ts_input = ts_input[sort_inds]
    ts_target = ts_target[sort_inds]
    
    X_all = np.squeeze(hankel_matrix(ts_input, tau))
    y_all = ts_target[tau:]

    if model == "linear":
        model = LinearRegression()
    elif model == "ridge":
        model = RidgeCV()
    elif model == "lasso":
        model = LassoCV()
    elif model == "gp":
        model = GaussianProcessRegressor()
    elif model == "rf":
        model = RandomForestRegressor()
    elif model == "mlp":
        model = MLPRegressor()
    else:
        model = LinearRegression()
        
    if split is None:
        split_point = -1
#         model.fit(X_all, y_all)
#         y_train_predict = model.predict(X_all)
#         return y_train_predict
        X_train, X_test = X_all, X_all
        y_train, y_test = y_all, y_all
    else:
        split_point = int(len(ts_input) * split)
        X_train, X_test = X_all[:split_point], X_all[split_point:]
        y_train, y_test = y_all[:split_point], y_all[split_point:]
    

    
    model.fit(X_train, y_train)
    y_test_predict = model.predict(X_test)
    import matplotlib.pyplot as plt
#     plt.figure()
#     plt.plot(y_all)
#     plt.plot(np.arange(len(y_train)), y_train)
#     plt.plot(len(y_train) + np.arange(len(y_test)), y_test)
#     plt.plot(len(y_train) + np.arange(len(y_test)), y_test_predict)
    
#     plt.plot(y_test, y_test_predict, '.k')
    #plt.plot(X_test[:, 0], y_test_predict, '.k')
    
    y_test_predict,  y_test = standardize_ts(y_test_predict), standardize_ts(y_test)
    score = np.mean((y_test_predict - y_test)**2)
    score = np.abs(spearmanr(y_test_predict, y_test).correlation)
    if return_points:
        return score, y_test_predict
    else:
        return score


def cross_forecast_error(ts_reference, ts_predicted, tau_vals=[5, 10, 25, 50, 100], **kwargs):
    """
    Compute the lowest cross-forecast error across a range of lookback timescales, and 
    report the best
    """
    mse = np.inf
    corr = -np.inf
    for tau in tau_vals:
        prediction = cross_forecast(ts_predicted, ts_reference, 
                                    tau=tau, **kwargs)
        corr = max(corr,
                   np.abs(spearmanr(prediction, ts_reference[tau:]).correlation)
                  )
        mse = min(mse, np.mean((prediction - ts_reference[tau:])**2))
    return corr