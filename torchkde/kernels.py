from abc import ABC, abstractmethod
import math

import torch
from scipy.special import gamma

SUPPORTED_KERNELS = [
    "gaussian",
    "epanechnikov"
]


class Kernel(ABC):
    @abstractmethod
    def __call__(self, x):
        pass


class GaussianKernel(Kernel):
    def __init__(self, bandwidth):
        self.bandwidth = bandwidth

    def __call__(self, x):
        u = ((x / self.bandwidth)**2).sum(-1)
        return torch.exp(-u/2) / \
                ((2 * math.pi)**(x.shape[-1]*0.5))


class EpanechnikovKernel(Kernel):
    def __init__(self, bandwidth):
        self.bandwidth = bandwidth

    def __call__(self, x):
        c = self._norm_constant(dim=x.shape[-1])
        u = ((x / self.bandwidth)**2).sum(-1)
        return torch.where(u > 1, 0, c * (1 - u))
    
    def _norm_constant(self, dim):
        # normalizing constant for the Epanechnikov
        return ((dim + 2)*gamma(dim/2 + 1))/(2*math.pi**(dim/2))
     