#!/usr/bin/python
import numpy as np

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA

from elephant.spike_train_generation import inhomogeneous_poisson_process
from elephant.gpfa import GPFA
import quantities as pq
import neo


class GPFAContinuous:
    """
    A class for fitting a GPFA model to time series data. Each time series is treated
    as the rate function for an inhomogenous Poisson process.
    
    Attributes
        n_components (int): number of variables to fit
        latent_dimensionality (int): dimensionality of the latent space
        bin_size (int): size of the time bins to use for fitting
        num_trials (int): number of trials. We deliberately use a very large number, in
            order to minimize the amount of information lost when the signal is 
            converted to spikes.
    """


    def __init__(self, n_components=2, bin_size=1, latent_dimensionality=4, num_trials=50):
        self.n_components = n_components
        self.latent_dimensionality = latent_dimensionality
        self.bin_size = bin_size * pq.ms
        self.num_trials = num_trials

    def _numpy_to_spikes(self, X):
        """Convert a numpy array of time series to a list of spike trains."""
        timestep = pq.ms
        spiketrains = []
        for _ in range(self.num_trials):
            spiketrains_per_trial = []
            for signal in X.T:
                signal = MinMaxScaler().fit_transform(signal[:, None])
                anasig_inst_rate = neo.AnalogSignal(signal, sampling_rate=1 / timestep, units=pq.Hz)
                spiketrains_per_trial.append(inhomogeneous_poisson_process(anasig_inst_rate))
            spiketrains.append(spiketrains_per_trial)
        return spiketrains

    def fit(self, X):
        X_spike = self._numpy_to_spikes(X)
        self.model_gpfa = GPFA(bin_size=self.bin_size, x_dim=self.latent_dimensionality)
        self.model_gpfa.fit(X_spike)
        return self

    def transform(self, X):
        X_spike = self._numpy_to_spikes(X)
        out = self.model_gpfa.fit_transform(X_spike)
        out = np.dstack(out)
        out = np.transpose(out, (1, 0, 2))
        out = np.reshape(out, (len(out), -1))
        out = StandardScaler().fit_transform(out)
        return PCA().fit_transform(out)[:, :self.n_components]

    def fit_transform(self, X):
        X_spike = self._numpy_to_spikes(X)
        self.model_gpfa = GPFA(bin_size=self.bin_size, x_dim=self.latent_dimensionality)
        out = self.model_gpfa.fit_transform(X_spike)
        out = np.dstack(out)
        out = np.transpose(out, (1, 0, 2))
        out = np.reshape(out, (len(out), -1))
        out = StandardScaler().fit_transform(out)
        return PCA().fit_transform(out)[:, :self.n_components]