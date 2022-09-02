#!/usr/bin/python
import numpy as np
import os
from scipy.signal import resample
import itertools

from gpfa import GPFAContinuous

model_name = "gpfa"
OUTPUT_DIR = './dump_results'
SEED = 0
latent_dim_vals = [1, 5, 10, 20]
num_trial_vals = [5, 10, 20, 40, 80]

# train_data_name = "eco" # pick which dataset to use
# train_data_name = "rat" # pick which dataset to use
# train_data_name = "fluid" # pick which dataset to use
train_data_name = "ecg" # pick which dataset to use
DATA_DIR = './data'

# use train/val split for all tuning
# train_data = np.load(os.path.join(DATA_DIR, train_data_name + '_data_val.npy'), allow_pickle=True)
# true_signal = np.load(os.path.join(DATA_DIR, train_data_name + '_driver_val.npy'), allow_pickle=True)
# oracle-benchmarks for unsupervised learning---make the baselines more competitive
train_data = np.load(os.path.join(DATA_DIR, train_data_name + '_data.npy'), allow_pickle=True)
true_signal = np.load(os.path.join(DATA_DIR, train_data_name + '_driver.npy'), allow_pickle=True)
train_data = train_data[:]
true_signal = true_signal[:train_data.shape[0]]



model_name = "gpfa"
all_params = itertools.product(num_trial_vals, latent_dim_vals)
fname_base = "all_signals_" + model_name +  "_" + train_data_name + "_"
for params in all_params:

#     ## Everything here is model specific
    trial_val, dim_val = params
    print("hyperparameters: ", trial_val, dim_val, flush=True)

    # Sometimes bad hyperparameters trigger a NaN loss
    try:
        #model = DCA(d=dim_val, T=tau, rng_or_seed=SEED)
        model = GPFAContinuous(
            n_components=1, num_trials=trial_val, latent_dimensionality=dim_val
        )
        y_model = model.fit_transform(train_data).squeeze()
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

