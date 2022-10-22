import copy
import matplotlib.pyplot as plt
import numpy as np
import torch

# Random Fourier features mapping
def map(x, B):
    if B is None:
        return x
    else:
        return torch.cat([torch.sin(2 * np.pi * x @ B.t()), torch.cos(2 * np.pi * x @ B.t())], dim = -1)

# Ground truth solution
def phi(omega, x):
    return torch.sin(omega * x).prod(dim = -1).unsqueeze(1)

# Plot numerical solutions and ground truth
def plot(hparams, models, B, res = 300):
    x1, x2 = np.meshgrid(np.linspace(-np.pi, np.pi, res), np.linspace(-np.pi, np.pi, res))
    xs = torch.from_numpy(np.hstack([x1.reshape(-1, 1), x2.reshape(-1, 1), np.pi / (2 * hparams['omega']) * np.ones((res ** 2, hparams['dim'] - 2))]))
    phis = phi(hparams['omega'], xs).reshape(res, res)

    N = [None] * len(models)

    for order in range(len(models)):
        model = models['EC' + str(order)]
        N[order] = model(map(xs.float(), B)).detach().numpy().reshape(res, res)

        if order > 0:
            N[order] += N[order - 1]

    fig, axes = plt.subplots(1, len(models) + 1, figsize = (4 * (len(models) + 2), 4))

    for j, ax in enumerate(axes.flat):
        if ax == axes[-1]:
            p = ax.pcolormesh(x1, x2, phis, cmap = 'viridis', shading = 'auto', vmin = -1, vmax = 1)
            ax.set_title('Ground Truth')
        else:
            p = ax.pcolormesh(x1, x2, N[j], cmap = 'viridis', shading = 'auto', vmin = -1, vmax = 1)
            ax.set_title(f'Error Correction {j}')
            rel_error = torch.sum(np.square(phis - N[j])) / torch.sum(np.square(phis))
            ax.set_xlabel(f'Relative error: {rel_error:.3f}')

    fig.colorbar(p, ax = axes.ravel().tolist())

# Plot training metrics
def plot_metrics(hparams, metrics):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (12, 8))
    start = 0

    for order in range(len(hparams['epochs'])):
        end = start + 100 * int(hparams['epochs'][order] / 100)
        epochs = np.arange(start, end + 100, 100)
        start = end
        colour = [float(order) / float(len(hparams['epochs'])), 0.0, float(len(hparams['epochs']) - order) / float(len(hparams['epochs']))]
        ax1.plot(epochs, metrics['EC' + str(order) + '-losses'], color = colour, label = f'ORDER {order}')
        ax2.plot(epochs, metrics['EC' + str(order) + '-errors'], color = colour, label = f'ORDER {order}')

    ax1.legend(loc = 'upper right', ncol = len(hparams['epochs']), fontsize = 'xx-small')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_yscale('log')

    ax2.legend(loc = 'upper right', ncol = len(hparams['epochs']), fontsize = 'xx-small')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Relative error')
    ax2.set_yscale('log')

# Train network
def train(dataloader, device, model, models, optimiser, order, B, loss_fn = torch.nn.MSELoss()):
    best_loss = np.inf
    best_model = copy.deepcopy(model)

    losses = np.zeros(int(dataloader.dataset.epochs / 100) + 1)
    errors = np.zeros_like(losses)

    for epoch, (X, is_boundary) in enumerate(dataloader):
        X, is_boundary = X.to(device), is_boundary.to(device)
        X.requires_grad = True

        # Forward pass
        N = model(map(X, B))
        Ns = torch.zeros(len(N), order + 1)

        # Forward pass through lower order correction models
        for j in range(order):
            Ns[:, j + 1] = models['EC' + str(j)](map(X, B)).squeeze()

        # Calculate relative error
        if epoch % 100 == 0:
            with torch.no_grad():
                error = torch.sum(torch.square(phi(dataloader.dataset.omega, X) - N - torch.sum(Ns, dim = 1, keepdim = True))) / torch.sum(torch.square(phi(dataloader.dataset.omega, X)))

        # Calculate loss
        F = dataloader.dataset.diffeq(order, N, Ns, X)
        loss = loss_fn(F, torch.zeros_like(F)) + torch.mean(torch.square(N[is_boundary] + torch.sum(Ns, dim = 1, keepdim = True)[is_boundary] - 0)) # -boundary value

        # Save model with smallest loss
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_model = copy.deepcopy(model)

        # Save metrics
        if epoch % 100 == 0:
            print(f'epoch {epoch} \t loss: {loss.item():.6f} \t error: {error:.6f}')
            losses[int(epoch / 100)] = loss.item()
            errors[int(epoch / 100)] = error

        # Backward pass
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    return best_model, losses, errors
