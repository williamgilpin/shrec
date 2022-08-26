#!/usr/bin/python

import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, Dataset

from causal_cnn import CausalConvolutionBlock


class Autoencoder(nn.Module):
    """
    A causal dilated autoencoder for time series

    Attributes:
        latent_size (int): The size of the latent space

    """
    def __init__(self, latent_size=8, input_channels=1, kernel_size=3, latent_width=16):
        super().__init__()
        self.latent_size = latent_size
        self.input_channels = input_channels
        self.kernel_size = kernel_size
        self.latent_width = latent_width
        dilation = 2

        self.encoder = nn.Sequential(
            # nn.Conv1d(1, 16, kernel_size=5, stride=3, padding=2),  # b, 16, 10, 10
            CausalConvolutionBlock(self.input_channels, self.latent_width, self.kernel_size, 2, final=False),
            #             CausalConvolutionBlock(16, 8, 3, 2, final=False),
            nn.Conv1d(self.latent_width, self.latent_size, kernel_size=self.kernel_size, stride=3, padding=1),  # b, 8, 3, 3
        )

        padding = (self.kernel_size - 1) * dilation
        kk = int(padding / dilation + 1)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(
                self.latent_size, self.latent_width, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size // 2
            ),  # b, 16, 5, 5
            nn.ELU(True),
            nn.ConvTranspose1d(
                self.latent_width, 8, kernel_size=self.kernel_size, stride=3, padding=self.kernel_size // 2,
            ),  # b, 8, 15, 15
            nn.ELU(True),
            nn.ConvTranspose1d(
                8, self.input_channels, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size // 2 
            ),  # b, 1, 28, 28
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x