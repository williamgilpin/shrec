import numpy as np
import warnings

import dtw
from sktime.performance_metrics.forecasting import mean_absolute_scaled_error

from scipy.stats import spearmanr
from shrec.utils import standardize_ts

# from mutual_info import mutual_information_scaled

def sliding_score(y_true, y_pred, func, padding=0.3):
    """Compute a sliding score between two time series.

    Args:
        y_true (np.array): true time series
        y_pred (np.array): predicted time series
        func (function): function to compute the score
        padding (float): proportion of padding to use

    Returns:
        np.array: sliding score
    """
    y_true, y_pred = y_true.squeeze(), y_pred.squeeze()
    scores = list()
    max_len = min(len(y_true), len(y_pred))
    padding = int(padding * max_len)
    y_true, y_pred = y_true[:max_len], y_pred[:max_len]
    for i in range(0, max_len - padding):
        # correlate the true and predicted signals for each time offset
        n_max = len(y_pred[i:])
        scores.append(func(y_true[0: n_max], y_pred[i:]))
    return np.array(scores).squeeze()

def spearman_sliding(x, y):
    """Compute the sliding spearman distance"""
    spearman_func = lambda x, y: 1 - np.abs(spearmanr(x, y).correlation)
    sliding_scores = min(sliding_score(x, y, spearman_func))
    return sliding_scores

def mse_sliding(x, y):
    """Compute the sliding mean squared error"""
    x, y = standardize_ts(x), standardize_ts(y)
    mse_func = lambda x, y: np.mean((x - y)**2)
    sliding_scores1 = min(sliding_score(x, y, mse_func))
    sliding_scores2 = min(sliding_score(x, -y, mse_func))
    return min(sliding_scores1, sliding_scores2)

def smape_sliding(x, y):
    """Compute the sliding symmetric mean absolute percentage error"""
    x, y = standardize_ts(x), standardize_ts(y)
    smape_func = lambda x, y: np.mean(2 * np.abs(x - y) / (np.abs(x) + np.abs(y)))
    sliding_scores1 = min(sliding_score(x, y, smape_func))
    sliding_scores2 = min(sliding_score(x, -y, smape_func))
    return min(sliding_scores1, sliding_scores2)

def dtw_sliding(x, y):
    """Compute the sliding dynamic time warping distance"""
    x, y = standardize_ts(x), standardize_ts(y)
    out1 = dtw.dtw(x, y).distance
    out2 = dtw.dtw(x, -y).distance
    return min(out1, out2)

# def mutual_information_sliding(x, y):
#     x, y = standardize_ts(x), standardize_ts(y)
#     mi_func = lambda x, y: 1 - mutual_information_scaled(x, y, k=30)
#     sliding_scores1 = min(sliding_score(x, y, mi_func))
#     sliding_scores2 = min(sliding_score(x, -y, mi_func))
#     return min(sliding_scores1, sliding_scores2)


def mase_sliding(x, y):
    """Compute the sliding mean absolute scaled error"""
    x, y = standardize_ts(x), standardize_ts(y)
    mase_func = lambda x, y: mean_absolute_scaled_error(x, y, y_train=x)
    sliding_scores1 = min(sliding_score(x, y, mase_func))
    sliding_scores2 = min(sliding_score(x, -y, mase_func))
    return min(sliding_scores1, sliding_scores2)
