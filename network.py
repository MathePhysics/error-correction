import numpy as np
import torch

# Sine activation
class Sine(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.sin(input)

# Siren weights initialisation scheme
def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        m.weight.data.uniform_(-np.sqrt(6 / m.in_features), np.sqrt(6 / m.in_features))
        m.bias.data.fill_(0)

# Siren
def Siren(hparams):
    depth, width = hparams['depth'], hparams['width']

    # Fourier features
    if hparams['num_features']:
        dim = 2 * hparams['num_features']
    else:
        dim = hparams['dim']

    layers = [torch.nn.Linear(dim, width), Sine()]

    for i in range(1, depth - 1):
        layers.append(torch.nn.Linear(width, width))
        layers.append(Sine())

    layers.append(torch.nn.Linear(width, 1))

    return torch.nn.Sequential(*layers).apply(init_weights)
