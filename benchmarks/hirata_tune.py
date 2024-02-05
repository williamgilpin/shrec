#!/usr/bin/python
import numpy as np
import os

import sys
sys.path.append('../') ## Local import code from this repo
from shrec.models import HirataNomuraIsomap

model_name = "hirata"
OUTPUT_DIR = './dump_results'
SEED = 0
percentile_vals = [0.001, 0.01, 0.1, 0.2]

# train_data_name = "gene" # pick which dataset to use
# train_data_name = "eco" # pick which dataset to use
# train_data_name = "monkey" # pick which dataset to use
# train_data_name = "fluid" # pick which dataset to use
train_data_name = "ecg" # pick which dataset to use
DATA_DIR = './data'

## If no output directory, make it
if not os.path.exists(os.path.join(OUTPUT_DIR, model_name, train_data_name)):
    os.makedirs(os.path.join(OUTPUT_DIR, model_name, train_data_name))

# use train/val split for all tuning
train_data = np.load(os.path.join(DATA_DIR, train_data_name + '_data.npy'), allow_pickle=True)
true_signal = np.load(os.path.join(DATA_DIR, train_data_name + '_driver.npy'), allow_pickle=True)
train_data = train_data[:]
true_signal = true_signal[:train_data.shape[0]]

all_params = percentile_vals
fname_base = "all_signals_" + model_name +  "_" + train_data_name + "_"
for params in all_params:

#     ## Everything here is model specific
    percentile = params
    print("hyperparameters: ", percentile, flush=True)

    # Sometimes bad hyperparameters trigger a NaN loss
    try:
        model = HirataNomuraIsomap(n_components=1, percentile=percentile_vals[0], random_state=SEED)
        y_model = model.fit_transform(train_data).squeeze()
        signal = y_model#[:train_data.shape[0]]
        print(f"Finished training {model_name}.", flush=True)
        pstr = str(percentile)
        signal.dump(
            os.path.join(
                OUTPUT_DIR, model_name, train_data_name, fname_base + pstr + '.npy'
            )
        )
    except ValueError:
        print("Error with hyperparameters: ", percentile, flush=True)
        pass

