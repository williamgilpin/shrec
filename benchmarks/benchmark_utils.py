import numpy as np
import warnings
import pandas as pd
import warnings
import numpy as np

import warnings

from scipy.signal import coherence
from scipy.stats import spearmanr, pearsonr, kendalltau, wilcoxon

from darts import TimeSeries
import darts.metrics.metrics
import dtw; # suppress stdout

from shrec.utils import *

import os

# Hack to solve relative import problems
import os
import sys
cwd = os.path.dirname(os.path.realpath(__file__))
sys.path.append(cwd)
from mutual_info import mutual_information, conditional_information
from phase_synchrony import sync_average

def print_dict(a):
    for row in a:
        print(np.round(a[row], 2), row, flush=True)
    print("\n")

class DrivenLorenz:
    """
    An ensemble of Lorenz systems driven by an external signal

    Parameters
        random_state (int): seed for the random number generator
        driver (str): type of driver, either "rossler" or "periodic"
    """
    def __init__(self, random_state=None, driver="rossler"):
        
        # driver properties
        self.ad = 0.5
        self.n = 5.3
        self.r = 1

        if driver == "rossler":
            self.rhs_driver = self._rhs_rossler
        elif driver == "periodic":
            self.rhs_driver= self._rhs_periodic
        else:
            warnings.warn("Unknown driver type, defauling to Rossler")
            self.rhs_driver= self._rhs_rossler
        
        # response properties
#         self.ar = 1.2
#         self.mu = 8.53
#         self.w = 0.63
        self.rho = 28
        self.beta = 2.667
        self.sigma = 10
        
        self.n_drive = 3
        self.n_response = 3
        
        ## rossler
        self.a = 0.2
        self.b = 0.2
        self.c = 5.7
        
        np.random.seed(random_state)
        self.n_sys = 24
        self.rho = 28 * (1 + 0.5*(np.random.random(self.n_sys) - 0.5))
        self.beta = 2.667 * (1 + 0.1*(np.random.random(self.n_sys) - 0.5))
        self.sigma = 10 * (1 + 0.1*(np.random.random(self.n_sys) - 0.5))
        

        self.n_sys = 12 * 3 * 2 * 4
        self.rho = 28 * (1 + 1 + 0.5*(np.random.random(self.n_sys) - 0.5))
        self.beta = 2.667 * (1 + 0.1*(np.random.random(self.n_sys) - 0.5))
        self.sigma = 10 * (1 + 0.1*(np.random.random(self.n_sys) - 0.5))
        
        
        
        self.rho = 28 * (1 + 1 + 0.1 * (np.random.random(self.n_sys) - 0.5))
        self.beta = 2.667 * (1 + 0.1 * (np.random.random(self.n_sys) - 0.5))
        self.sigma = 10 * (1 + 2 * (np.random.random(self.n_sys) - 0.5))
        
        
        self.rho = 28 * (1 + 5 * (np.random.random(self.n_sys)))
        self.beta = 2.667 * (1 +  0.1 * (np.random.random(self.n_sys)))
        self.sigma = 10 * (1 + 20 * (np.random.random(self.n_sys) ))
        
        
        
        self.rho = 28 * (1 + 2 * 5 * (np.random.random(self.n_sys)))
        self.beta = 2.667 * (1 +  2 * 1 * (np.random.random(self.n_sys)))
        self.sigma = 10 * (1 + 2 * 10 * (np.random.random(self.n_sys) ))
        

    def _rhs_periodic(self, t, X):
        """Simple periodic driver"""
        x, y, z = X
        a, n, r = self.ad, self.n, self.r
        xdot = a * 15 * np.sin(t / 2) - x
        ydot =  a * 15 * np.sin(t / 2) - y
        zdot =  a * 15 * np.sin(t / 2) - z
        return xdot, ydot, zdot

    def _rhs_rossler(self, t, X):
        """Rossler driving (aperiodic)"""
        x, y, z = X
        a, b, c = self.a, self.b, self.c
        xdot = -y - z
        ydot = x + a * y
        zdot = b + z * (x - c)
        return xdot * 0.5, ydot * 0.5, zdot * 0.5
    
    def rhs_response_ensemble(self, t, X):
        """Response system

        Args:
            t (float): time
            X (np.ndarray): state vector

        Returns:
            np.ndarray: derivative of the state vector
        """
        
        Xd = X[:self.n_drive]
        Xr = X[self.n_drive:]
        
        xd, yd, zd = Xd
        x, y, z = Xr[:self.n_sys], Xr[self.n_sys:2*self.n_sys], Xr[2 * self.n_sys:]

        xdot = self.sigma * (y - x) + self.ar * xd
        ydot = x * (self.rho - z) - y # - self.ar * xd
        zdot = x * y - self.beta * z
        return np.hstack([xdot, ydot, zdot])
    
    def rhs(self, t, X):
        """Full system

        Args:
            t (float): time
            X (np.ndarray): state vector

        Returns:
            np.ndarray: derivative of the state vector
        """
        return [*self.rhs_driver(t, X[:self.n_drive]), *self.rhs_response_ensemble(t, X)]


def fft_phase(a):
    """Compute the phase of the FFT of a signal"""
    ns = len(a)
    af = np.fft.fft(a)
    af = af / np.abs(af)
    af = af[:ns // 2]
    angs = np.arctan2(np.imag(af), np.real(af))
    return angs
     
def coherence_phase(a, b, freq=1, FS=1):
    """Compute the phase of the coherence between two signals"""
    ap, bp = fft_phase(a), fft_phase(b)
    return np.sqrt((np.cos(ap) + np.cos(bp))**2 + (np.sin(ap) + np.sin(bp))**2) / 2

def score_ts2(true_y, pred_y):
    """ Score a pair of time series"""
    true_yn, pred_yn = (
        np.squeeze(standardize_ts(true_y)), 
        np.squeeze(standardize_ts(pred_y))
    )


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
    
    scores["cross forecast error"] = cross_forecast(pred_y, true_y, split=0.75)
    #scores["cross forecast error rf"] = cross_forecast(pred_y, true_y, model="rf", split=0.75)
    scores["cross forecast error neural"] = cross_forecast(pred_y, true_y, model="mlp", split=0.75)
    
    scores["dynamic time warping distance"] = min(dtw.dtw(pred_y, true_y).normalizedDistance, dtw.dtw(-pred_y, true_y).normalizedDistance)

    return scores


from statsmodels.tsa.stattools import grangercausalitytests

def granger_f(y_true, y_pred):
    """
    Compute the f statistic for a Granger causality test between two time series
    """
    y_true, y_pred = y_true.squeeze(), y_pred.squeeze()

    # exception for constant time series
    if np.all(y_true == y_true[0]) or np.all(y_pred == y_pred[0]):
        return 1.0e-16

    # Use statsmodels rule for max lag for adf test
    max_lag = int(12 * (len(y_true) / 100)**(1/4))

    try:
        gran = grangercausalitytests(np.vstack([y_pred, y_true]).T, maxlag=max_lag, verbose=False)
    except:
        return 1.0e-16
    max_f = max([gran[item][0]["ssr_ftest"][0] for item in gran])
    return max_f

from sktime.performance_metrics.forecasting import mean_absolute_scaled_error

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from statsmodels.tsa.stattools import adfuller

def score_ts(true_yn, pred_yn):
    """ Score a pair of time series"""

    true_yn, pred_yn = np.squeeze(true_yn), np.squeeze(pred_yn)
    true_yn, pred_yn = detrend_ts(true_yn).squeeze(), detrend_ts(pred_yn).squeeze()

    # with warnings.catch_warnings():
    #     warnings.filterwarnings("ignore")
    #     # Difference if non-stationary
    #     if (adfuller(true_yn)[1] > 0.05) or (adfuller(pred_yn)[1] > 0.05):
    #         true_yn, pred_yn = np.diff(true_yn), np.diff(pred_yn)
    
    true_yn, pred_yn = np.squeeze(true_yn), np.squeeze(pred_yn)
    #true_yn, pred_yn = MinMaxScaler().fit_transform(true_yn[:, None]) + 1e-4, MinMaxScaler().fit_transform(pred_yn[:, None]) + 1e-4
    #true_yn, pred_yn = StandardScaler().fit_transform(true_yn[:, None]), StandardScaler().fit_transform(pred_yn[:, None])
    true_yn, pred_yn = (
        RobustScaler().fit_transform(true_yn[:, None]), 
        RobustScaler().fit_transform(pred_yn[:, None])
    )
    true_yn, pred_yn = np.squeeze(true_yn), np.squeeze(pred_yn)
    
    
    # Functional dependence
    # sort_inds = np.argsort(true_yn)
    # true_yn, pred_yn = true_yn[sort_inds], pred_yn[sort_inds]
    # true_yn, pred_yn = np.cumsum(true_yn), np.cumsum(pred_yn)
    
    scores = dict()
    
    kval = 30
    np.random.seed(0)
    lo  = min([
        mutual_information((
            np.random.permutation(true_yn)[:, None],
            np.random.permutation(true_yn)[:, None]),
            k=kval
        )
        for _ in range(20)
    ])
    hi = mutual_information((true_yn[:, None], true_yn[:, None]), k=kval)
    mi = mutual_information((true_yn[:, None], pred_yn[:, None]), k=kval)
    scores["one minus mutual information"] = 1 - (mi - lo) / (hi - lo)
    #scores["conditional_info"] = conditional_information(true_yn, pred_yn, k=kval)
    #scores["conditional_info_back"] = conditional_information(pred_yn, true_yn, k=kval)
    
    scores["mase"] = min(
        mean_absolute_scaled_error(true_yn, pred_yn, y_train=true_yn),
        mean_absolute_scaled_error(true_yn, -pred_yn, y_train=true_yn),
    )
    
    # ## Special format for darts metrics
    # metric_list = [
    #     'mae',
    #     'marre',
    #     'mse',
    #     'rmse',
    #     'smape'
    # ]
    # true_y_df = TimeSeries.from_dataframe(pd.DataFrame(np.squeeze(true_yn)))
    # pred_y_df = TimeSeries.from_dataframe(pd.DataFrame(np.squeeze(pred_yn)))
    # pred_y_df_neg = TimeSeries.from_dataframe(pd.DataFrame(np.squeeze(-pred_yn)))
    # for metric_name in metric_list:
    #     metric_func = getattr(darts.metrics.metrics, metric_name)
    #     try:
    #         if metric_name in ['r2_score']:
    #             scores[metric_name] = max(
    #                 metric_func(true_y_df, pred_y_df), 
    #                 metric_func(true_y_df, pred_y_df_neg)
    #             )
    #         else:
    #             scores[metric_name] = min(
    #                 metric_func(true_y_df, pred_y_df), 
    #                 metric_func(true_y_df, pred_y_df_neg)
    #             )
    #     except:
    #         print(metric_name, " Skipped")
    
    # corr = spearmanr(true_yn, pred_yn)
    # scores["one minus spearman"] = 1 - np.abs(corr.correlation)
    # corr = pearsonr(true_yn, pred_yn)[0]
    # scores["one minus pearson"] = 1 - np.abs(corr)
    # corr = kendalltau(true_yn, pred_yn)[0]
    # scores["one minus kendalltau"] = 1 - np.abs(corr)

    
    # scores["one minus sync"] = 1 - max(sync_average(true_yn, pred_yn), sync_average(true_yn, -pred_yn))
    # scores["one minus coherence"] = 1 - np.mean(coherence(true_yn, pred_yn)[1])
    # scores["one minus coherence_phase"] = 1 - np.mean(coherence_phase(true_yn, pred_yn)[1])

    scores["granger_f_inv"] = 1 / (1.0e-16 + granger_f(true_yn, pred_yn))
    
    scores["cross forecast error"] = cross_forecast(pred_yn, true_yn)
    scores["cross forecast error neural"] = cross_forecast(pred_yn, true_yn, model="mlp")
    scores["cross forecast error neural 2"] = cross_forecast(true_yn, pred_yn, model="mlp")
    scores["cross forecast error gradboost"] = cross_forecast(pred_yn, true_yn, model="gb")
    scores["cross forecast error gradboost 2"] = cross_forecast(true_yn, pred_yn, model="gb")

    # scores["dynamic time warping distance"] = min(
    #     dtw.dtw(pred_yn, true_yn).normalizedDistance, 
    #     dtw.dtw(-pred_yn, true_yn).normalizedDistance
    # )
    
    return scores

from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR

def cross_forecast(
    ts_input, ts_target, 
    tau=50,
    score="granger",
    model="ridge", 
    split=0.7,
    return_points=False
):
    """
    Train a forecast model that predicts a target time series using timepoints
    of a given time series. 

    Args:
        ts_input (np.ndarray): The input time series of shape (N, D)
        ts_target (np.ndarray): The target time series of shape (N, D)
        tau (int): The number of timepoints to use for prediction
        model (str): The model to use. Can be "ridge", "lasso", "gp", "rf", "mlp"
        split (float): the fraction of the data to use as a test split
        return_points (bool): Whether to return the points used for training
        return_model (bool): Whether to return the trained model

    Returns:
        score (float): The cross forecast error
        y_test_predict (np.ndarray): The predicted values

    
    Uses Wilcoxon statistic as a measure of coupling and causality between the two
    signals. Wilcoxon is used becase the f-test assumptions do not hold for nonlinear
    time series models
    Citation: https://www.sciencedirect.com/science/article/pii/S0169260722000542
    
    """
    
    ts_input, ts_target = np.squeeze(detrend_ts(ts_input)), np.squeeze(detrend_ts(ts_target))
    if (adfuller(ts_input)[1] > 0.05) or (adfuller(ts_target)[1] > 0.05):
            ts_input, ts_target = np.diff(ts_input), np.diff(ts_target)

    # Find a functional mapping based on values of the input time series
    # sort_inds = np.argsort(ts_input)
    # ts_input = ts_input[sort_inds]
    # ts_target = ts_target[sort_inds]
    
    #y_all = ts_target[tau:]

    ## Augmented data
    X_all = np.squeeze(hankel_matrix(ts_input, tau))
    Y_all = np.squeeze(hankel_matrix(ts_target, tau))
    y_all = ts_target[tau:]
    X_all_aug = np.concatenate([X_all, Y_all], axis=1)

    if model == "linear":
        model = LinearRegression()
    elif model == "ridge":
        model = RidgeCV()
    elif model == "svr":
        model = SVR(kernel="rbf")
    elif model == "kridge":
        model = KernelRidge(kernel="rbf", alpha=1e-1)
    elif model == "lasso":
        model = LassoCV()
    elif model == "gp":
        model = GaussianProcessRegressor()
    elif model == "rf":
        model = RandomForestRegressor()
    elif model == "gb":
        model = HistGradientBoostingRegressor()
    elif model == "mlp":
        # wider networks more consistent
        model = MLPRegressor(hidden_layer_sizes=(500, 500)) 
    else:
        model = LinearRegression()
        
    if split is None:
        split_point = -1
#         model.fit(X_all, y_all)
#         y_train_predict = model.predict(X_all)
#         return y_train_predict
        X_train, X_test = X_all, X_all
        X_train_aug, X_test_aug = X_all_aug, X_all_aug
        Y_train, Y_test = Y_all, Y_all
        y_train, y_test = y_all, y_all
    else:
        split_point = int(len(ts_input) * split)
        X_train, X_test = X_all[:split_point], X_all[split_point:]
        X_train_aug, X_test_aug = X_all_aug[:split_point], X_all_aug[split_point:]
        Y_train, Y_test = Y_all[:split_point], Y_all[split_point:]
        y_train, y_test = y_all[:split_point], y_all[split_point:]
    
    if score == "granger":
        # Generalized Granger causality
        # replace f statistic with wilcoxon test of whether the error is generally smaller
        # for the augmented model
        ## Fit restricted model
        model.fit(Y_train, y_train)
        y_predict = model.predict(Y_test)
        ## Fit unrestricted model
        model.fit(X_train_aug, y_train)
        y_predict_aug = model.predict(X_test_aug)
        mse_restricted = np.mean((y_predict - y_test)**2)
        mse_unrestricted = np.mean((y_predict_aug - y_test)**2)
        fstat = (mse_restricted - mse_unrestricted) / mse_unrestricted
        score = 1 / fstat # 1 / fstat a distance measure based on fstatistic
        #print(mse_unrestricted, mse_restricted, fstat)

        # larger wilcoxon test statistic means that the augmented model is better
        try:
            wstat = wilcoxon((y_predict - y_test)**2, (y_predict_aug - y_test)**2).statistic / len(y_test)**2
        except ValueError:
            wstat = 0.0
        score = 1 / (wstat + 1.0e-10) # 1 / wstat is a distance measure
        
    elif score == "aug_mse":
        # Augmented model
        model.fit(X_train_aug, y_train)
        y_predict_aug = model.predict(X_test_aug)
        score = np.mean((y_predict_aug - y_test)**2)

    #elif score == "cross_mse":
    else:
        warnings.warng("Score method not recognized. Using cross-forecast mse")
        # Cross-fitting
        model.fit(X_train, y_train)
        y_predict = model.predict(X_test)
        score = np.mean((y_predict - y_test)**2)



    # score = 1 - np.abs(spearmanr(y_test_predict, y_test).correlation)
    if return_points:
        return score, y_predict_aug
    else:
        return score

def cross_forecast_validate(ts_predicted, ts_reference, tau_vals=[5, 10, 25, 50, 100], **kwargs):
    """
    Choose the best time lag for a forecast model for Granger causality by validating 
    on a subset of the training data
    """
    test_index = int(0.7 * len(ts_reference))
    all_scores = []
    for tau in tau_vals:
        score = cross_forecast(
            ts_predicted, ts_reference, 
            tau=tau, 
            score = "granger",
            **kwargs
        )
        all_scores.append(score)
    print(all_scores, tau_vals)
    best_tau = tau_vals[np.argmin(all_scores)]
    print(best_tau)

    score = cross_forecast(
            ts_predicted, ts_reference, 
            tau=best_tau, 
            score="granger",
            **kwargs
        )

    return score

def cross_forecast_error(ts_reference, ts_predicted, tau_vals=[5, 10, 25, 50, 100], **kwargs):
    """
    Compute the lowest cross-forecast error across a range of lookback timescales, and 
    report the best

    Args:
        ts_reference (np.ndarray): The reference time series of shape (N, D)
        ts_predicted (np.ndarray): The predicted time series of shape (N, D)
        tau_vals (list): The list of lookback timescales to test

    Returns:
        corr (float): The best cross-forecast error   
    """
    mse = np.inf
    score = -np.inf
    for tau in tau_vals:
        prediction = cross_forecast(ts_predicted, ts_reference, tau=tau, **kwargs)
        corr = max(corr,
                   np.abs(spearmanr(prediction, ts_reference[tau:]).correlation)
                  )
        mse = min(mse, np.mean((prediction - ts_reference[tau:])**2))
    return corr

