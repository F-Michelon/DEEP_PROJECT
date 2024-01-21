import numpy as np
import tqdm

import torch
from torchvision import datasets, transforms

from sklearn.model_selection import KFold
from sklearn.neighbors import KernelDensity

def log_likelihood(data, samples, h):
    '''
    Compute the log-likelihood of the data given the samples
    '''
    # Compute the log-likelihood of the data
    kde = KernelDensity(kernel='gaussian', bandwidth=h).fit(samples)
    log_likelihood = kde.score(data)
 
    return np.mean(log_likelihood)

if __name__ == '__main__':

    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    mnist = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    data = mnist.data.numpy().reshape(-1, 28*28)

    # Load Gan model
    cgan = torch.load('saved_models/baseline_cgan.pth')

    # Generate 10000 samples
    num_samples =  data.shape[0]
    samples = []
    for i in range(num_samples):
        z = torch.randn(1, 100)
        sample = cgan.generator(z, torch.tensor([0]))
        samples.append(sample.detach().numpy().reshape(-1))
    samples = np.array(samples)

    # Compute log-likelihood
    ll = log_likelihood(data, samples, h=.1)
    print('log-likelihood:', ll)