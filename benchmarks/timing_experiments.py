#!/usr/bin/python
import os, sys
import time
import numpy as np
import json

# 
SEED = 2

from baseline_models import FourierPCA, CCATimeSeries
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from scipy.signal import resample

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader 
device = 'cuda' if torch.cuda.is_available() else 'cpu'; print('Using device: %s'%device)

import sktime.transformations.series.kalman_filter as kf

import sys
sys.path.append('../') ## Local import code from this repo
from shrec.utils import *
from shrec.models import *

import jax.numpy as jnp

train_data_name = "eco" # pick which dataset to use

# train_data_name = "gene" # pick which dataset to use
# train_data_name = "rat" # pick which dataset to use
# train_data_name = "fluid" # pick which dataset to use
# train_data_name = "ecg" # pick which dataset to use
DATA_DIR = './data'
train_data = np.load(os.path.join(DATA_DIR, train_data_name + '_data.npy'), allow_pickle=True)

output_path = f"timing_results_{train_data_name}_{SEED}.json"
## if exists, load the file
if os.path.exists(output_path):
    with open(output_path, "r") as f:
        all_timing_results = json.load(f)
else:
    all_timing_results = dict()


if all_timing_results.get("fPCA") is not None:
    print("Already computed fPCA")
else:
    hparam = 0.1
    model = FourierPCA(n_components=2, sig_thresh=hparam)
    time_start = time.perf_counter()
    model.fit(train_data)
    time_end = time.perf_counter()
    fit_time = str(time_end - time_start)
    time_start = time.perf_counter()
    y_model = model.transform(train_data)
    time_end = time.perf_counter()
    predict_time = str(time_end - time_start)
    all_timing_results["fPCA"] = {"fit_time": fit_time, "predict_time": predict_time}

if all_timing_results.get("ICA") is not None:
    print("Already computed ICA")
else:
    model = FastICA(n_components=2, random_state=SEED)
    time_start = time.perf_counter()
    model.fit(train_data)
    time_end = time.perf_counter()
    fit_time = str(time_end - time_start)
    time_start = time.perf_counter()
    y_model = model.transform(train_data)
    time_end = time.perf_counter()
    predict_time = str(time_end - time_start)
    all_timing_results["ICA"] = {"fit_time": fit_time, "predict_time": predict_time}

if all_timing_results.get("SFA") is not None:
    print("Already computed SFA")
else:
    from sksfa import SFA
    model = SFA(n_components=2, random_state=SEED)
    time_start = time.perf_counter()
    model.fit(train_data)
    time_end = time.perf_counter()
    fit_time = str(time_end - time_start)
    time_start = time.perf_counter()
    y_model = model.transform(train_data)
    time_end = time.perf_counter()
    predict_time = str(time_end - time_start)
    all_timing_results["SFA"] = {"fit_time": fit_time, "predict_time": predict_time}

if all_timing_results.get("CCA") is not None:
    print("Already computed CCA")
else:
    model = CCATimeSeries(n_components=2, time_lag=20)
    time_start = time.perf_counter()
    model.fit(train_data)
    time_end = time.perf_counter()
    fit_time = str(time_end - time_start)
    time_start = time.perf_counter()
    y_model = model.transform(train_data)
    time_end = time.perf_counter()
    predict_time = str(time_end - time_start)
    all_timing_results["CCA"] = {"fit_time": fit_time, "predict_time": predict_time}


if all_timing_results.get("PCA") is not None:
    print("Already computed PCA")
else:
    model = PCA(n_components=2)
    time_start = time.perf_counter()
    model.fit(train_data)
    time_end = time.perf_counter()
    fit_time = str(time_end - time_start)
    time_start = time.perf_counter()
    y_model = model.transform(train_data)
    time_end = time.perf_counter()
    predict_time = str(time_end - time_start)
    all_timing_results["PCA"] = {"fit_time": fit_time, "predict_time": predict_time}

if all_timing_results.get("DCA") is not None:
    print("Already computed DCA")
else:
    from dca import DynamicalComponentsAnalysis as DCA
    model = DCA(d=2, T=20, rng_or_seed=SEED)
    time_start = time.perf_counter()
    model.fit(train_data)
    time_end = time.perf_counter()
    fit_time = str(time_end - time_start)
    time_start = time.perf_counter()
    y_model = model.transform(train_data)
    time_end = time.perf_counter()
    predict_time = str(time_end - time_start)
    all_timing_results["DCA"] = {"fit_time": fit_time, "predict_time": predict_time}
    print("DCA", flush=True)

if all_timing_results.get("Mean") is not None:
    print("Already computed Mean")
else:
    time_start = time.perf_counter()
    np.mean(train_data, axis=1)
    time_end = time.perf_counter()
    fit_time = str(time_end - time_start)
    time_start = time.perf_counter()
    y_mean = np.mean(train_data, axis=1)
    time_end = time.perf_counter()
    predict_time = str(time_end - time_start)
    all_timing_results["Mean"] = {"fit_time": fit_time, "predict_time": predict_time}
    print("Mean", flush=True)

if all_timing_results.get("Kalman") is not None:
    print("Already computed Kalman")
else:
    model = kf.KalmanFilterTransformerFP(state_dim=1)
    time_start = time.perf_counter()
    model.fit(train_data)
    time_end = time.perf_counter()
    fit_time = str(time_end - time_start)
    time_start = time.perf_counter()
    y_model = model.transform(train_data)
    time_end = time.perf_counter()
    predict_time = str(time_end - time_start)
    all_timing_results["Kalman"] = {"fit_time": fit_time, "predict_time": predict_time}
    print("Kalman", flush=True)

if all_timing_results.get("shrec") is not None:
    print("Already computed shrec")
else:
    model = RecurrenceManifold()
    time_start = time.perf_counter()
    model.fit_predict(train_data)
    time_end = time.perf_counter()
    fit_time = str(time_end - time_start)
    time_start = time.perf_counter()
    model.labels_
    time_end = time.perf_counter()
    predict_time = str(time_end - time_start)
    all_timing_results["shrec"] = {"fit_time": fit_time, "predict_time": predict_time}
    print("shrec", flush=True)

if all_timing_results.get("GPFA") is not None:
    print("Already computed GPFA")
else:
    from gpfa import GPFAContinuous
    trial_val, dim_val = 10, 2
    model = GPFAContinuous(
        n_components=2, num_trials=trial_val, latent_dimensionality=dim_val
    )
    time_start = time.perf_counter()
    model.fit(train_data)
    time_end = time.perf_counter()
    fit_time = str(time_end - time_start)
    time_start = time.perf_counter()
    y_model = model.transform(train_data)
    time_end = time.perf_counter()
    predict_time = str(time_end - time_start)
    all_timing_results["GPFA"] = {"fit_time": fit_time, "predict_time": predict_time}
    print("GPFA", flush=True)

if all_timing_results.get("LG-SSM") is not None:
    print("Already computed LG-SSM")
else:
    from dynamax.linear_gaussian_ssm import lgssm_smoother, parallel_lgssm_smoother
    from dynamax.linear_gaussian_ssm import LinearGaussianSSM
    model = LinearGaussianSSM(5, train_data.shape[1])
    emissions = jnp.array(train_data)
    params2, _ = model.initialize()
    # ssm_posterior = lgssm_smoother(params2, emissions)
    time_start = time.perf_counter()
    parallel_posterior = parallel_lgssm_smoother(params2, emissions)
    time_end = time.perf_counter()
    fit_time = str(time_end - time_start)
    time_start = time.perf_counter()
    y_model = np.array(parallel_posterior.filtered_means)[:, 0]
    time_end = time.perf_counter()
    predict_time = str(time_end - time_start)
    all_timing_results["LG-SSM"] = {"fit_time": fit_time, "predict_time": predict_time}
    print("LG-SSM", flush=True)


if all_timing_results.get("Hirata") is not None:
    print("Already computed Hirata")
else:
    from shrec.models import HirataNomuraIsomap
    percentile_val = 0.1
    model = HirataNomuraIsomap(n_components=1, percentile=0.1, random_state=SEED)
    time_start = time.perf_counter()
    model.fit(train_data)
    time_end = time.perf_counter()
    fit_time = str(time_end - time_start)
    time_start = time.perf_counter()
    y_model = model.labels_
    time_end = time.perf_counter()
    predict_time = str(time_end - time_start)
    all_timing_results["Hirata"] = {"fit_time": fit_time, "predict_time": predict_time}
    print("Hirata", flush=True)


if all_timing_results.get("cCNN") is not None:
    print("Already computed cCNN")
else:
    from causal_encoder import Autoencoder
    train_data2 = np.load(os.path.join(DATA_DIR, train_data_name + '_data.npy'), allow_pickle=True)
    train_data2 = train_data2[:crop_ind]
    input = torch.tensor(train_data2.T[None, ...], dtype=torch.float32)
    hyperparams = {
        'learning_rate': 1e-4,
        'latent_dim': 1,
    }
    lr, wd, fd = 1e-2, 32, 5
    model = Autoencoder(fd, input.shape[1], latent_width=wd)
    train_dataloader = DataLoader(input, batch_size=1, shuffle=True) # only one sample  
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    time_start = time.perf_counter()
    for epoch in range(200): 
        running_loss = 0.0
        for i, data_batch in enumerate(train_dataloader):
            #print(data_batch.shape)
            
            inputs = data_batch
    #         #print(inputs.shape)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(inputs, outputs)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    time_end = time.perf_counter()
    fit_time = str(time_end - time_start)
    time_start = time.perf_counter()
    pred = model.encoder(input).detach().numpy()[0, 0, :]
    signal = resample(pred, input.shape[-1])
    time_end = time.perf_counter()
    predict_time = str(time_end - time_start)
    all_timing_results["cCNN"] = {"fit_time": fit_time, "predict_time": predict_time}
    print("Autoencoder done", flush=True)


if all_timing_results.get("LFADS") is not None:
    print("Already computed LFADS")
else:
    from lfads import LFADS_Net
    train_data2 = torch.Tensor(np.array([train_data]).T).to(device)
    num_trials, num_steps, num_cells = train_data2.shape
    train_ds = torch.utils.data.TensorDataset(train_data2)
    hyperparams = {
        'betas': (0.9, 0.99), # Adam optimizer hyperparameters
        'c_encoder_dim': 128,
        'clip_val': 5.0,
        'controller_dim': 128,
        'dataset_name': 'test',
        'epsilon': 0.1, # Adam optimizer hyperparameter
        'factors_dim': 20,
        'g0_encoder_dim': 200,
        'g0_prior_kappa': 0.1,
        'g_dim': 20,
        'keep_prob': 0.95,
        'kl_weight_schedule_dur': 2000,
        'kl_weight_schedule_start': 0,
        'l2_con_scale': 0,
        'l2_gen_scale': 2000,
        'l2_weight_schedule_dur': 2000,
        'l2_weight_schedule_start': 0,
        'learning_rate': 1e-4,
        'learning_rate_decay': 0.95,
        'learning_rate_min': 1e-05,
        'max_norm': 200,
        'run_name': 'demo',
        'scheduler_cooldown': 6,
        'scheduler_on': True,
        'scheduler_patience': 6,
        'u_dim': 1,
        'u_prior_kappa': 0.1
    }
    lr, bs, fd = 1e-2, 20, 5
    model = LFADS_Net(inputs_dim = num_cells, T = num_steps, dt = 0.01, device=device,
                    model_hyperparams=hyperparams, seed=SEED).to(device)
    time_start = time.perf_counter()
    model.fit(train_ds, train_ds, max_epochs=5, batch_size=bs, use_tensorboard=False)
    time_end = time.perf_counter()
    fit_time = str(time_end - time_start)
    model.load_checkpoint('best')
    time_start = time.perf_counter()
    signal = model.reconstruct(train_data2)
    time_end = time.perf_counter()
    predict_time = str(time_end - time_start)
    all_timing_results["LFADS"] = {"fit_time": fit_time, "predict_time": predict_time}

## Save to JSON
import json
with open(output_path, "w") as f:
    json.dump(all_timing_results, f, indent=4)