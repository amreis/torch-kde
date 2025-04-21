from abc import ABC, abstractmethod
import math

import torch
from scipy.special import gamma, iv

from .utils import check_if_mat, inverse_sqrt


SUPPORTED_KERNELS = [
    "gaussian",
    "epanechnikov",
    "exponential",
    "tophat-approx",
    "von-mises-fisher"
]


class Kernel(ABC):
    def __init__(self):
        self._bandwidth = None
        self._norm_constant = None
        self.dim = None

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
    
    @property
    def norm_constant(self):
        if self._norm_constant is None:
            assert self.dim is not None, "Dimension not set."
            self._norm_constant = self._compute_norm_constant(self.dim)
        return self._norm_constant

    @abstractmethod
    def _compute_norm_constant(self, dim):
        pass

    @abstractmethod
    def __call__(self, x1, x2):
        assert self.bandwidth is not None, "Bandwidth not set."


class GaussianKernel(Kernel):
    def __call__(self, x1, x2):
        super().__call__(x1, x2)
        differences = x1 - x2
        self.dim = differences.shape[-1]
        u = kernel_input(self.inv_bandwidth, differences)

        return torch.exp(-u/2)
    
    def _compute_norm_constant(self, dim):
        if check_if_mat(self._bandwidth):
            # When bandwidth is a matrix, include sqrt(det(bandwidth))
            bw_norm = torch.sqrt(torch.det(self._bandwidth))
        else:
            # When bandwidth is a scalar, raise it to the dim/2
            bw_norm = self._bandwidth**(dim/2)
        return 1 / ((2 * math.pi)**(dim/2) * bw_norm)


class TopHatKernel(Kernel):
    """Differentiable approximation of the top-hat kernel 
    via a generalized Gaussian."""
    def __init__(self, beta=8):
        super().__init__()
        assert type(beta) == int, "beta must be an integer."
        self.beta = beta

    def __call__(self, x1, x2):
        super().__call__(x1, x2)
        differences = x1 - x2
        self.dim = differences.shape[-1]
        u = kernel_input(self.inv_bandwidth, differences)

        return torch.exp(-(u**self.beta)/2)
    
    def _compute_norm_constant(self, dim):
        if check_if_mat(self._bandwidth):
            # When bandwidth is a matrix, include sqrt(det(bandwidth))
            bw_norm = torch.sqrt(torch.det(self._bandwidth))
        else:
            # When bandwidth is a scalar, raise it to the d/2
            bw_norm = self._bandwidth**(dim/2)
        return (self.beta*gamma(dim/2))/(math.pi**(dim/2) * \
                                         gamma(dim/(2*self.beta)) * 2**(dim/(2*self.beta)) * bw_norm)


class EpanechnikovKernel(Kernel):
    def __init__(self):
        super().__init__()
        self._intrinsic_norm_constant = None

    def __call__(self, x1, x2):
        super().__call__(x1, x2)
        differences = x1 - x2
        self.dim = differences.shape[-1]
        c = self.intrinsic_norm_constant
        u = kernel_input(self.inv_bandwidth, differences)

        return torch.where(u > 1, 0, c * (1 - u))
    
    def _compute_intrinsic_norm_constant(self, dim):
        return ((dim + 2)*gamma(dim/2 + 1))/(2*math.pi**(dim/2))

    @property
    def intrinsic_norm_constant(self):
        """Return the cached intrinsic normalization constant, computing it if necessary."""
        if self._intrinsic_norm_constant is None:
            self._intrinsic_norm_constant = self._compute_intrinsic_norm_constant(self.dim)
        return self._intrinsic_norm_constant
    
    def _compute_norm_constant(self, dim):
        if check_if_mat(self._bandwidth):
            # When bandwidth is a matrix, include sqrt(det(bandwidth))
            bw_norm = torch.sqrt(torch.det(self._bandwidth))
        else:
            # When bandwidth is a scalar, raise it to the dim/2
            bw_norm = self._bandwidth**(dim/2)
        return 1 / bw_norm


class ExponentialKernel(Kernel):
    def __call__(self, x1, x2):
        super().__call__(x1, x2)
        differences = x1 - x2
        self.dim = differences.shape[-1]
        u = kernel_input(self.inv_bandwidth, differences, exp=1)

        return torch.exp(-u)
    
    def _compute_norm_constant(self, dim):
        if check_if_mat(self._bandwidth):
            # When bandwidth is a matrix, include sqrt(det(bandwidth))
            bw_norm = torch.sqrt(torch.det(self._bandwidth))
        else:
            # When bandwidth is a scalar, raise it to the dim/2
            bw_norm = self._bandwidth**(dim/2)
        return 1/(2**dim * bw_norm)
    

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


class VonMisesFisherKernel(Kernel):
    @Kernel.bandwidth.setter
    def bandwidth(self, bandwidth):
        # For vMF, the bandwidth is directly the concentration parameter.
        if type(bandwidth) == torch.Tensor:
            assert bandwidth.requires_grad == False, \
                "The bandwidth for the von Mises-Fisher kernel must not require gradients."
        assert type(bandwidth) == float or isinstance(bandwidth, torch.Tensor) and bandwidth.dim() == 0, \
            "The bandwidth for the von Mises-Fisher kernel must be a scalar."
        self._bandwidth = bandwidth
        self.inv_bandwidth = bandwidth

    def __call__(self, x1, x2):
        super().__call__(x1, x2)
        x_all = torch.cat([x1, x2], dim=1)
        assert torch.allclose(
            x_all.norm(dim=-1), torch.ones_like(x_all[..., 0]), atol=1e-5
        ), "The von Mises-Fisher kernel assumes all data to lie on the unit sphere. Please normalize data."
        self.dim = x1.shape[-1]
        
        return torch.exp(self._bandwidth * (x1 * x2).sum(dim=-1))

    def _compute_norm_constant(self, dim):
        assert not check_if_mat(self._bandwidth), "The von Mises-Fisher kernel only support scalar bandwidth arguments."
        # normalizing constant for the vMF kernel. Reference, e.g.: https://www.jmlr.org/papers/volume6/banerjee05a/banerjee05a.pdf
        return (self._bandwidth**(dim/2 - 1))/((2*math.pi)**(dim/2) * float(iv(dim/2 - 1, self._bandwidth)))
    