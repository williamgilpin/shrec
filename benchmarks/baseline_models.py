import numpy as np
# import scipy
import warnings

from shrec.utils import *


RANDOM_SEED = 0





## CCA

from sklearn.cross_decomposition import CCA

class CCATimeSeries(CCA):
    """
    Canonical Correlation Analysis for time series data.

    Parameters
        n_components (int): number of components to keep
        time_lag (int): number of time steps to lag the data
        cca (sklearn.cross_decomposition.CCA): CCA object used internally
        upsample (bool): whether to upsample the data to the same length as the input
        pad (bool): whether to pad the data to the same length as the input
    """
    def __init__(self, n_components=2, time_lag=1, pad=True):
        self.n_components = n_components
        self.time_lag = time_lag
        self.pad = pad

        self.cca = CCA(n_components=n_components)

    def fit(self, X):
        """
        Fit the model with X.

        Args:
            X (np.ndarray): Time series data of shape (n_timepoints, n_features)
        """
        Y = X[self.time_lag:]
        X = X[:-self.time_lag]
        self.cca.fit(X, Y)
        return self

    def transform(self, X):
        Y = X[self.time_lag:]
        X = X[:-self.time_lag]
        Xc, _ = self.cca.transform(X, Y)
        #print(Xc.shape, Yc.shape)
        if self.pad:
            Xc = np.vstack([Xc, Xc[-self.time_lag:][::-1]])
        return Xc


## Fourier

from sklearn.base import BaseEstimator, TransformerMixin

class FourierPCA(BaseEstimator, TransformerMixin):
    """
    Given a collection of time series, average their power spectra and then combine
    their phase time series using PCA. Can optionally use surrogate time series to
    discard insignificant frequencies.

    Parameters
        sig_thresh (float): threshold for discarding insignificant frequencies
        n_components (int): number of components to keep in PCA
        random_state (int): random state for use when sampling surrogates
    """
    def __init__(self, n_components=1, sig_thresh=0.0, random_state=None):
        self.n_components = n_components
        self.sig_thresh = sig_thresh
        self.random_state = random_state

    def fit(self, X=None, y=None):
        """Placeholder for sklearn compatibility"""
        return self

    def transform(self, X, y=None):
        """
        Transform a stack of input time series into a stack of Fourier-PCA features
        
        Args:
            X (np.ndarray): time series data, shape (n_samples, n_timepoints)

        Returns:
            np.ndarray: transformed data, shape (n_samples, n_components)
        """
        
        all_amps, all_phases = list(), list()
        for i in range(X.shape[1]):
            fft0 = np.fft.fft(X[:, i])
            amps, phases = np.abs(fft0), np.angle(fft0)

            if self.sig_thresh > 0:
                surr = make_surrogate(X[:, i], ns=500, random_state=self.random_state).T
                surr_fft = np.fft.fft(surr)
                surr_amps = np.abs(surr_fft)
                surr_amps_upper = np.percentile(surr_amps, 100 * self.sig_thresh, axis=0)
                amps[amps < surr_amps_upper] = 0

            all_amps.append(amps)
            all_phases.append(phases)

        amps_ave = np.median(np.array(all_amps), axis=0)[:, None]
        phases_ave = PCA().fit_transform(np.array(all_phases).T)[:, :self.n_components]

        fft_ave = amps_ave * np.cos(phases_ave) + 1j * amps_ave * np.sin(phases_ave)
        sig = np.fft.ifft(fft_ave, axis=0).real.squeeze()
        return sig



