#!/usr/bin/python
import numpy as np
import os
import itertools

import jax.numpy as jnp
from dynamax.linear_gaussian_ssm import lgssm_smoother, parallel_lgssm_smoother
from dynamax.linear_gaussian_ssm import LinearGaussianSSM

model_name = "lgssm"
OUTPUT_DIR = './dump_results'
SEED = 0
latent_dim_vals = [2, 5, 10, 20]

# train_data_name = "gene" # pick which dataset to use
train_data_name = "eco" # pick which dataset to use
# train_data_name = "monkey" # pick which dataset to use
# train_data_name = "fluid" # pick which dataset to use
# train_data_name = "ecg" # pick which dataset to use
DATA_DIR = './data'

# use train/val split for all tuning
# train_data = np.load(os.path.join(DATA_DIR, train_data_name + '_data_val.npy'), allow_pickle=True)
# true_signal = np.load(os.path.join(DATA_DIR, train_data_name + '_driver_val.npy'), allow_pickle=True)
train_data = np.load(os.path.join(DATA_DIR, train_data_name + '_data.npy'), allow_pickle=True)
true_signal = np.load(os.path.join(DATA_DIR, train_data_name + '_driver.npy'), allow_pickle=True)
train_data = train_data
true_signal = true_signal[:train_data.shape[0]]

all_params = itertools.product(latent_dim_vals)
fname_base = "all_signals_" + model_name +  "_" + train_data_name + "_"
observation_dim = train_data.shape[1]
emissions = jnp.array(train_data)
print(train_data.shape, flush=True)
for params in all_params:

    ## Everything after here is model specific
    latent_dim = params[0]
    print("hyperparameters: ", *params, flush=True)

    # Sometimes bad hyperparameters trigger a NaN loss
    try:
        model = LinearGaussianSSM(latent_dim, observation_dim)
        params2, _ = model.initialize()
        ssm_posterior = lgssm_smoother(params2, emissions)
        parallel_posterior = parallel_lgssm_smoother(params2, emissions)
        y_model = np.array(parallel_posterior.filtered_means)[:, 0]

        signal = y_model#[:train_data.shape[0]]
        print(f"Finished training {model_name}.", flush=True)
        pstr = "_".join([str(x) for x in params])
        signal.dump(
            os.path.join(
                OUTPUT_DIR, model_name, train_data_name, fname_base + pstr + '.npy'
            )
        )
    except ValueError:
        print("Error with hyperparameters: ", *params, flush=True)
        pass





# import jax.numpy as jnp
# import jax.random as jr
# import matplotlib.pyplot as plt
# from dynamax.hidden_markov_model import GaussianHMM

# key1, key2, key3 = jr.split(jr.PRNGKey(0), 3)
# num_states = 20
# emission_dim = X_measure.shape[1]
# # num_timesteps = 1000

# # Make a Gaussian HMM and sample data from it
# # hmm = GaussianHMM(num_states, emission_dim)
# # true_params, _ = hmm.initialize(key1)
# # true_states, emissions = hmm.sample(true_params, key2, num_timesteps)

# emissions = X_measure

# # Make a new Gaussian HMM and fit it with EM
# hmm = GaussianHMM(num_states, emission_dim)
# params, props = hmm.initialize(key3, method="kmeans", emissions=emissions)
# params, lls = hmm.fit_em(params, props, emissions, num_iters=20)

# # Plot the marginal log probs across EM iterations
# plt.plot(lls)
# plt.xlabel("EM iterations")
# plt.ylabel("marginal log prob.")

# # Use fitted model for posterior inference
# post = hmm.smoother(params, emissions)
# print(post.smoothed_probs.shape) # (1000, 3)


# # reconstruction
# out = np.array(post.smoothed_probs)
# estimated_states = np.argmax(out, axis=1)

# from dysts.utils import standardize_ts
# plt.figure()
# plt.plot(standardize_ts(estimated_states[:, None]))
# plt.plot(standardize_ts(y_signal))