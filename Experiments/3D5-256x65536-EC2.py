"""
This script is a template to run experiments with error correction.

The following hyperparameters may be set in the hparams dictionary:
    alpha: probability of point being sampled from boundary
    batch_size: number of points in a single pass
    depth: number of hidden layers in network
    dim: number of spatial dimensions
    epochs: list of number of batch passes per error correction
    lr: learning rate
    num_features: number of Fourier features
    omega: frequency of solution
    scale_features: Fourier features scale
    width: number of hidden units per layer
"""

import sys
sys.path.append('.')

from data import *
from network import *
from train import *

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    hparams = {}
    hparams['alpha'] = 0.1
    hparams['batch_size'] = 2 ** 8
    hparams['depth'] = 5
    hparams['dim'] = 3
    hparams['epochs'] = [2 ** 14, 2 ** 14, 2 ** 15]
    hparams['lr'] = 0.005
    hparams['num_features'] = 0
    hparams['omega'] = 5
    hparams['scale_features'] = 1
    hparams['width'] = 64

    # Fourier features
    if hparams['num_features']:
        B = (torch.randn((hparams['num_features'], hparams['dim'])) * hparams['scale_features']).to(device)
    else:
        B = None

    metrics = {}
    models = {}

    for order in range(len(hparams['epochs'])):
        dataset = Poisson(hparams, hparams['epochs'][order])
        dataloader = DataLoader(dataset, dataset.batch_size)
        model = Siren(hparams).to(device)
        optimiser = torch.optim.Adam(model.parameters(), lr = 2 ** (-1 * order) * hparams['lr'])

        print(f'ORDER {order}')

        best_model, losses, errors = train(dataloader, device, model, models, optimiser, order, B)
        metrics['EC' + str(order) + '-losses'] = losses
        metrics['EC' + str(order) + '-errors'] = errors
        models['EC' + str(order)] = best_model

    filename = f"{hparams['dim']}D{hparams['omega']}-{hparams['batch_size']}x{sum(hparams['epochs'])}-EC{len(hparams['epochs']) - 1}"

    plot(hparams, models, B)
    plt.savefig(filename + '.png', bbox_inches = 'tight')

    plot_metrics(hparams, metrics)
    plt.savefig(filename + '-metrics.png', bbox_inches = 'tight')

if __name__ == '__main__':
    main()
