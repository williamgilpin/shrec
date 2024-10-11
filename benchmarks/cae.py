
import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader 

from causal_encoder import Autoencoder

class CausalAutoencoder:
    """
    A causal autoencoder that learns a causal driver from a set of time series

    Attributes:
        lr (float): The learning rate
        wd (int): The width of the encoder and decoder
        fd (int): The width of the latent space
        model (Autoencoder): The autoencoder model
        criterion (nn.MSELoss): The loss function
        optimizer (optim.Adam): The optimizer
    """
    def __init__(self, lr=1e-4, wd=16, fd=1):
        self.lr = lr
        self.wd = wd
        self.fd = fd
        self.model = None
        self.criterion = nn.MSELoss()
        self.optimizer = None

    def fit(self, X):
        input = torch.tensor(X.T[None, ...], dtype=torch.float32)
        self.model = Autoencoder(self.fd, input.shape[1], latent_width=self.wd)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        train_dataloader = DataLoader(input, batch_size=1, shuffle=True) # only one sample
        for epoch in range(200): 
            running_loss = 0.0
            for i, data_batch in enumerate(train_dataloader):
                inputs = data_batch
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(inputs, outputs)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
        print("Finished training autoencoder.", flush=True)
        return self

    def predict(self, X):
        input = torch.tensor(X.T[None, ...], dtype=torch.float32)
        return self.model.encoder(input).detach().numpy()[0, 0, :]
    
    def fit_transform(self, X):
        self.fit(X)
        return self.predict(X)