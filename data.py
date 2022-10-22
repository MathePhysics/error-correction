import numpy as np
import torch

from torch.utils.data import TensorDataset, DataLoader

class Poisson(TensorDataset):
    """
    This class samples points upon which a neural network is trained to solve Poisson's equation.

    The methods work as follows:
        __init__:
            Initialises class attributes.
            Randomly samples points from a hypercube with vertex coordinates of either -pi or pi.
            Marks which points are on the boundary, i.e. have at least one coordinate -pi or pi.
        __getitem__:
            Returns point and Boolean value describing boundary-ness by given index.
        __len__:
            Returns number of points (batch_size x epochs) in dataset.
        diffeq:
            Poisson's equation in operator form and recursively defined error corrections.
        f:
            RHS of Poisson's equation.
    """

    def __init__(self, hparams, epochs):
        self.alpha = hparams['alpha']
        self.batch_size = hparams['batch_size']
        self.dim = hparams['dim']
        self.epochs = epochs
        self.omega = hparams['omega']

        X = -2 * np.pi * torch.rand(self.batch_size * self.epochs, self.dim) + np.pi
        is_boundary = torch.from_numpy(np.random.choice([False, True], size = X.shape, p = [1 - self.alpha, self.alpha]))
        X[is_boundary] = np.random.choice([-np.pi, np.pi])

        self.is_boundary = torch.any(is_boundary, dim = -1, keepdim = True)
        self.X = X

    def __getitem__(self, idx):
        return self.X[idx], self.is_boundary[idx]

    def __len__(self):
        return len(self.X)

    def diffeq(self, order, N, Ns, x):
        if order == 0:
            return Laplacian(N, x) - self.f(x)
        else:
            return self.diffeq(order - 1, Ns[:, -1].unsqueeze(1), Ns[:, :-1], x) + Laplacian(N, x)

    def f(self, x):
        return (-(self.omega ** 2) * self.dim * torch.sin(self.omega * x).prod(dim = -1)).unsqueeze(1)

# Network Laplacian
def Laplacian(N, x):
    dX = torch.autograd.grad(N, x, grad_outputs = torch.ones_like(N), create_graph = True)[0]
    Laplacian = torch.zeros_like(N)

    for i in range(dX.shape[1]):
        dx = dX[:, i].reshape(-1, 1)
        dxx = torch.autograd.grad(dx, x, grad_outputs = torch.ones_like(dx), create_graph = True)[0][:, i].reshape(-1, 1)
        Laplacian += dxx

    return Laplacian
