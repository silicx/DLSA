import torch
import torch.nn as nn
import logging
import numpy as np
from math import pi
from tqdm import tqdm


class MLP(nn.Module):
    """Multi-layer perceptron, 1 layers as default. No activation after last fc"""

    def __init__(self, inp_dim, out_dim, hidden_layers=[], batchnorm=True, bias=True, out_relu=False):
        super(MLP, self).__init__()

        inner_bias = bias and (not batchnorm)

        mod = []
        if hidden_layers is not None:
            last_dim = inp_dim
            for hid_dim in hidden_layers:
                mod.append(nn.Linear(last_dim, hid_dim, bias=inner_bias))
                if batchnorm:
                    mod.append(nn.BatchNorm1d(hid_dim))
                mod.append(nn.ReLU(inplace=True))
                last_dim = hid_dim

            mod.append(nn.Linear(last_dim, out_dim, bias=bias))
            if out_relu:
                mod.append(nn.ReLU(inplace=True))

        self.mod = nn.Sequential(*mod)

    def forward(self, x):
        output = self.mod(x)
        return output