from abc import ABC, abstractmethod
import math

import torch
from scipy.special import gamma

from torchkde.utils import check_if_mat, inverse_sqrt


SUPPORTED_KERNELS = [
    "gaussian",
    "epanechnikov",
    "exponential"
]


class Kernel(ABC):
    def __init__(self):
        self._bandwidth = None

    @property
    def bandwidth(self):
        return self._bandwidth
    
    @bandwidth.setter
    def bandwidth(self, bandwidth):
        self._bandwidth = bandwidth
        # compute H^(-1/2)
        if check_if_mat(bandwidth):
            self.inv_bandwidth = inverse_sqrt(bandwidth)
        else:  # Scalar case
            self.inv_bandwidth = self.bandwidth**(-0.5)

    @abstractmethod
    def __call__(self, x):
        assert self.bandwidth is not None, "Bandwidth not set."


class GaussianKernel(Kernel):
    def __call__(self, x):
        super().__call__(x)
        c = self._norm_constant(dim=x.shape[-1])
        u = kernel_input(self.inv_bandwidth, x)
        return c*torch.exp(-u/2)

    def _norm_constant(self, dim):
        # normalizing constant for the Gaussian kernel
        return 1/(2*math.pi)**(dim/2)


class EpanechnikovKernel(Kernel):
    def __call__(self, x):
        super().__call__(x)
        c = self._norm_constant(dim=x.shape[-1])
        u = kernel_input(self.inv_bandwidth, x)
        return torch.where(u > 1, 0, c * (1 - u))
    
    def _norm_constant(self, dim):
        # normalizing constant for the Epanechnikov
        return ((dim + 2)*gamma(dim/2 + 1))/(2*math.pi**(dim/2))


class ExponentialKernel(Kernel):
    def __call__(self, x):
        super().__call__(x)
        c = self._norm_constant(dim=x.shape[-1])
        u = kernel_input(self.inv_bandwidth, x, exp=1)
        return c*torch.exp(-u)
    
    def _norm_constant(self, dim):
        # normalizing constant for the exponential kernel
        return 1/(2**dim)
    

def kernel_input(inv_bandwidth, x, exp=2):
    """Compute the input to the kernel function."""
    if exp >= 2:
        if check_if_mat(inv_bandwidth):
            return ((x @ inv_bandwidth)**exp).sum(-1)
        else:  # Scalar case
            return ((x * inv_bandwidth)**exp).sum(-1)
    else: # absolute value
        if check_if_mat(inv_bandwidth):
            return ((x @ inv_bandwidth).abs()).sum(-1)
        else:  # Scalar case
            return ((x * inv_bandwidth).abs()).sum(-1)
    