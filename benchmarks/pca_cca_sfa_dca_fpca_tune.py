#!/usr/bin/python
import numpy as np
import os
from scipy.signal import resample
import itertools

from baseline_models import FourierPCA, CCATimeSeries
from dca import DynamicalComponentsAnalysis as DCA
from sksfa import SFA
# from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA


OUTPUT_DIR = './dump_results'
SEED = 0
tau_vals = np.array([1, 5, 10, 20, 50, 100])
latent_dim_vals = [1, 5, 10, 20, 50]

tau_vals = [1, 5, 10, 20]
latent_dim_vals = [1, 5, 10]

# train_data_name = "gene" # pick which dataset to use
train_data_name = "eco" # pick which dataset to use
# train_data_name = "rat" # pick which dataset to use
# train_data_name = "fluid" # pick which dataset to use
# train_data_name = "ecg" # pick which dataset to use
DATA_DIR = './data'

# use train/val split for all tuning
# train_data = np.load(os.path.join(DATA_DIR, train_data_name + '_data_val.npy'), allow_pickle=True)
# true_signal = np.load(os.path.join(DATA_DIR, train_data_name + '_driver_val.npy'), allow_pickle=True)
# oracle-benchmarks for unsupervised learning---make the baselines more competitive
train_data = np.load(os.path.join(DATA_DIR, train_data_name + '_data.npy'), allow_pickle=True)
true_signal = np.load(os.path.join(DATA_DIR, train_data_name + '_driver.npy'), allow_pickle=True)
train_data = train_data[:]
true_signal = true_signal[:train_data.shape[0]]

## Avoid using hyperparameters that are too large for the dataset
# tau_vals = tau_vals[tau_vals < len(true_signal)]

latent_dim_vals[-1] = min(latent_dim_vals[-1], train_data.shape[1])

model_name = "dca"
all_params = itertools.product(tau_vals, latent_dim_vals)
fname_base = "all_signals_" + model_name +  "_" + train_data_name + "_"
for params in all_params:

    ## Everything here is model specific
    tau, dim_val = params
    print("hyperparameters: ", tau, dim_val, flush=True)

    # Sometimes bad hyperparameters trigger a NaN loss
    try:
        model = DCA(d=dim_val, T=tau, rng_or_seed=SEED)
        y_model = model.fit_transform(train_data)[:, 0]
        #signal = resample(y_model , input.shape[-1])
        signal = y_model#[:train_data.shape[0]]
        print(f"Finished training {model_name}.", flush=True)
        pstr = "_".join([str(x) for x in params])
        signal.dump(
            os.path.join(
                OUTPUT_DIR, model_name, train_data_name, fname_base + pstr + '.npy'
            )
        )
    except ValueError:
        print("Error with hyperparameters: ", dim_val, tau, flush=True)
        pass


model_name = "cca"
all_params = tau_vals
fname_base = "all_signals_" + model_name +  "_" + train_data_name + "_"
for params in all_params:

    ## Everything here is model specific
    tau = params
    print("hyperparameters: ", tau, flush=True)

    # Sometimes bad hyperparameters trigger a NaN loss
    try:
        model = CCATimeSeries(n_components=1, time_lag=tau)
        model.fit(train_data)
        y_model = model.transform(train_data)[:, 0]
        #signal = resample(y_model , input.shape[-1])
        signal = y_model[:train_data.shape[0]]
        print(f"Finished training {model_name}.", flush=True)
        pstr = str(params)
        signal.dump(
            os.path.join(
                OUTPUT_DIR, model_name, train_data_name, fname_base + pstr + '.npy'
            )
        )
    except ValueError:
        print("Error with hyperparameters: ", tau, flush=True)
        pass


model_name = "sfa"
all_params = latent_dim_vals
fname_base = "all_signals_" + model_name +  "_" + train_data_name + "_"
for params in all_params:

    ## Everything here is model specific
    dim = params
    print("hyperparameters: ", dim, flush=True)

    # Sometimes bad hyperparameters trigger a NaN loss
    try:
        model = SFA(n_components=dim, random_state=SEED)
        model.fit(train_data)
        y_model = model.transform(train_data)[:, 0]
        #signal = resample(y_model , input.shape[-1])
        signal = y_model[:train_data.shape[0]]
        print(f"Finished training {model_name}.", flush=True)
        pstr = str(params)
        signal.dump(
            os.path.join(
                OUTPUT_DIR, model_name, train_data_name, fname_base + pstr + '.npy'
            )
        )
    except ValueError:
        print("Error with hyperparameters: ", dim, flush=True)
        pass

model_name = "ica"
all_params = latent_dim_vals
fname_base = "all_signals_" + model_name +  "_" + train_data_name + "_"
for params in all_params:

    ## Everything here is model specific
    dim = params
    print("hyperparameters: ", dim, flush=True)

    # Sometimes bad hyperparameters trigger a NaN loss
    try:
        model = FastICA(n_components=dim, random_state=SEED)
        model.fit(train_data)
        y_model = model.transform(train_data)[:, 0]
        #signal = resample(y_model , input.shape[-1])
        signal = y_model[:train_data.shape[0]]
        print(f"Finished training {model_name}.", flush=True)
        pstr = str(params)
        signal.dump(
            os.path.join(
                OUTPUT_DIR, model_name, train_data_name, fname_base + pstr + '.npy'
            )
        )
    except ValueError:
        print("Error with hyperparameters: ", dim, flush=True)
        pass




model_name = "fpca"
all_params = np.linspace(0, 0.99, 10)
fname_base = "all_signals_" + model_name +  "_" + train_data_name + "_"
for params in all_params:

    ## Everything here is model specific
    thresh = params
    print("hyperparameters: ", thresh, flush=True)

    # Sometimes bad hyperparameters trigger a NaN loss
    try:
        model = FourierPCA(n_components=2, sig_thresh=thresh)
        model.fit(train_data)
        y_model = model.transform(train_data)[:, 0]
        #signal = resample(y_model , input.shape[-1])
        signal = y_model[:train_data.shape[0]]
        print(f"Finished training {model_name}.", flush=True)
        pstr = str(params)
        signal.dump(
            os.path.join(
                OUTPUT_DIR, model_name, train_data_name, fname_base + pstr + '.npy'
            )
        )
    except ValueError:
        print("Error with hyperparameters: ", dim, flush=True)
        pass
