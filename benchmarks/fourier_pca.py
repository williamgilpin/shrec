import numpy as np
import scipy
import warnings

from shrec.utils import *


RANDOM_SEED = 0



## Causal AE


## PCA


## DCA


## Fourier

import numpy as np
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

    def transform(self, X):
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



