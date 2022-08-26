#!/usr/bin/python
import numpy as np
import os
import sys

import torch
from lfads import LFADS_Net
device = 'cuda' if torch.cuda.is_available() else 'cpu'; print('Using device: %s'%device)

model_name = "lfads"
lr_vals = [1e-1, 1e-2, 1e-3]
batch_size_vals = [5, 20, 100]
factors_dim_vals = [5, 10, 20, 50]
OUTPUT_DIR = './dump_results'
SEED = 0

train_data_name = "rat" # pick which dataset to use
DATA_DIR = './data'

train_data = np.load(os.path.join(DATA_DIR, train_data_name + '_data_val.npy'), allow_pickle=True)
true_signal = np.load(os.path.join(DATA_DIR, train_data_name + '_driver_val.npy'), allow_pickle=True)
true_signal = true_signal[:train_data.shape[0]]

train_data = torch.Tensor(np.array([train_data]).T).to(device)
num_trials, num_steps, num_cells = train_data.shape
train_ds = torch.utils.data.TensorDataset(train_data)

# defaults
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
import itertools
all_params = itertools.product(lr_vals, batch_size_vals, factors_dim_vals)

data_path = os.path.join(DATA_DIR, train_data_name + '_val.npz')
fname_base = "all_signals_" + model_name +  "_" + train_data_name + "_"
for params in all_params:
    

    ## Everything here is model specific
    lr, bs, fd = params
    print("hyperparameters: ", lr, bs, fd, flush=True)
    hyperparams['learning_rate'] = lr
    hyperparams['factors_dim'] = fd

    # Sometimes bad hyperparameters trigger a NaN loss
    try:
        model = LFADS_Net(inputs_dim = num_cells, T = num_steps, dt = 0.01, device=device,
                        model_hyperparams=hyperparams, seed=SEED).to(device)
        # We are doing unsupervised learning, so we will use a single train/val dataset
        model.fit(train_ds, train_ds, max_epochs=5, batch_size=bs, use_tensorboard=False)
        model.load_checkpoint('best')
        signal = model.reconstruct(train_data)
        signal = np.squeeze(np.array(signal))
        pstr = "_".join([str(x) for x in params])
        signal.dump(os.path.join(OUTPUT_DIR, model_name, train_data_name, fname_base + pstr + '.npy'))
    except ValueErro:
        print("Error with hyperparameters: ", lr, bs, fd, flush=True)
        pass



    
    