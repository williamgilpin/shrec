#!/usr/bin/python
import numpy as np
import os
import sys
from scipy.signal import resample

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader 

from causal_encoder import Autoencoder

device = 'cuda' if torch.cuda.is_available() else 'cpu'; print('Using device: %s'%device)

model_name = "causal_cnn"
lr_vals = [1e-1, 1e-2, 1e-3]
width_vals = [8, 16, 32, 64]
latent_dim_vals = [1, 5, 10, 20, 50]
OUTPUT_DIR = './dump_results'
SEED = 0

train_data_name = "gene" # pick which dataset to use
# train_data_name = "eco" # pick which dataset to use
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
train_data = train_data[:2989]
true_signal = true_signal[:train_data.shape[0]]

input = torch.tensor(train_data.T[None, ...], dtype=torch.float32)

# defaults
hyperparams = {
    'learning_rate': 1e-4,
    'latent_dim': 1,
}
import itertools
all_params = itertools.product(lr_vals, width_vals, latent_dim_vals)

fname_base = "all_signals_" + model_name +  "_" + train_data_name + "_"
for params in all_params:

    ## Everything here is model specific
    lr, wd, fd = params
    print("hyperparameters: ", lr, wd, fd, flush=True)

    # Sometimes bad hyperparameters trigger a NaN loss
    try:
        # train_dataloader = DataLoader(input, batch_size=64, shuffle=True)
        train_dataloader = DataLoader(input, batch_size=1, shuffle=True) # only one sample

        model = Autoencoder(fd, input.shape[1], latent_width=wd)
        train_dataloader = DataLoader(input, batch_size=1, shuffle=True) # only one sample  

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-2)
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
        print("Finished training autoencoder.", flush=True)
        pred = model.encoder(input).detach().numpy()[0, 0, :]
        signal = resample(pred, input.shape[-1])


        pstr = "_".join([str(x) for x in params])
        signal.dump(os.path.join(OUTPUT_DIR, model_name, train_data_name, fname_base + pstr + '.npy'))
    except ValueError:
        print("Error with hyperparameters: ", lr, wd, fd, flush=True)
        pass
