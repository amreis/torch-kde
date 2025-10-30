from abc import ABC, abstractmethod
import math

import torch
from scipy.special import gamma, iv

from .utils import check_if_mat, inverse_sqrt


SUPPORTED_KERNELS = ["gaussian", "epanechnikov", "exponential", "tophat-approx", "von-mises-fisher"]


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
        self._norm_constant = None  # reset normalization constant when bandwidth changes
        self._bandwidth = bandwidth
        # compute H^(-1/2)
        if check_if_mat(bandwidth):
            self.inv_bandwidth = inverse_sqrt(bandwidth)
        else:  # Scalar case
            self.inv_bandwidth = self.bandwidth ** (-0.5)

    @property
    def norm_constant(self):
        if self._norm_constant is None:
            assert self.dim is not None, "Dimension not set."
            self._norm_constant = self._compute_norm_constant(self.dim)
        return self._norm_constant

    def _bw_norm(self, dim):
        if check_if_mat(self._bandwidth):
            return torch.sqrt(torch.det(self._bandwidth))
        else:
            return self._bandwidth ** (dim / 2)

    @abstractmethod
    def _compute_norm_constant(self, dim):
        pass

    @abstractmethod
    def __call__(self, x1, x2):
        """This __call__ must be called in child classes to clear the norm constant cache and check the inputs."""
        assert self.bandwidth is not None, "Bandwidth not set."
        assert x1.shape[-1] == x2.shape[-1], "Input data must have the same dimensionality."
        new_dim = x1.shape[-1]
        if new_dim != self.dim:  # first call or change of dimensionality
            self.dim = new_dim
            self._norm_constant = None


class GaussianKernel(Kernel):
    def __call__(self, x1, x2):
        super().__call__(x1, x2)
        differences = x1 - x2
        u = compute_u(self.inv_bandwidth, differences)

        return torch.exp(-u / 2)

    def _compute_norm_constant(self, dim):
        bw_norm = self._bw_norm(dim)
        return 1 / ((2 * math.pi) ** (dim / 2) * bw_norm)


class TopHatKernel(Kernel):
    """Differentiable approximation of the top-hat kernel
    via a generalized Gaussian."""

    def __init__(self, beta=8):
        super().__init__()
        assert isinstance(beta, int), "beta must be an integer."
        self.beta = beta

    def __call__(self, x1, x2):
        super().__call__(x1, x2)
        differences = x1 - x2
        u = compute_u(self.inv_bandwidth, differences)

        return torch.exp(-(u**self.beta) / 2)

    def _compute_norm_constant(self, dim):
        bw_norm = self._bw_norm(dim)
        return (self.beta * gamma(dim / 2)) / (
            math.pi ** (dim / 2)
            * gamma(dim / (2 * self.beta))
            * 2 ** (dim / (2 * self.beta))
            * bw_norm
        )


class EpanechnikovKernel(Kernel):
    def __init__(self):
        super().__init__()
        self._unit_ball_constant = None

    def __call__(self, x1, x2):
        old_dim = self.dim
        super().__call__(x1, x2)
        differences = x1 - x2
        if old_dim is not None and old_dim != differences.shape[-1]:
            self._unit_ball_constant = None
        c = self.unit_ball_constant
        u = compute_u(self.inv_bandwidth, differences)

        return torch.where(u > 1, 0, c * (1 - u))

    @Kernel.bandwidth.setter
    def bandwidth(self, bandwidth):
        Kernel.bandwidth.fset(self, bandwidth)
        self._unit_ball_constant = None  # reset the cached constant when bandwidth changes

    def _compute_unit_ball_constant(self, dim):
        return ((dim + 2) * gamma(dim / 2 + 1)) / (2 * math.pi ** (dim / 2))

    @property
    def unit_ball_constant(self):
        """Return the cached intrinsic normalization constant, computing it if necessary."""
        if self._unit_ball_constant is None:
            self._unit_ball_constant = self._compute_unit_ball_constant(self.dim)
        return self._unit_ball_constant

    def _compute_norm_constant(self, dim):
        bw_norm = self._bw_norm(dim)
        return 1 / bw_norm


class ExponentialKernel(Kernel):
    def __call__(self, x1, x2):
        super().__call__(x1, x2)
        differences = x1 - x2
        u = compute_u(self.inv_bandwidth, differences, exp=1)

        return torch.exp(-u)

    def _compute_norm_constant(self, dim):
        bw_norm = self._bw_norm(dim)
        return 1 / (2**dim * bw_norm)


def compute_u(inv_bandwidth, x, exp=2):
    """Compute the input to the kernel function."""
    if exp >= 2:
        if check_if_mat(inv_bandwidth):
            return ((x @ inv_bandwidth) ** exp).sum(-1)
        else:  # Scalar case
            return ((x * inv_bandwidth) ** exp).sum(-1)
    else:  # absolute value
        if check_if_mat(inv_bandwidth):
            return ((x @ inv_bandwidth).abs()).sum(-1)
        else:  # Scalar case
            return ((x * inv_bandwidth).abs()).sum(-1)


class VonMisesFisherKernel(Kernel):
    @Kernel.bandwidth.setter
    def bandwidth(self, bandwidth):
        Kernel.bandwidth.fset(self, bandwidth)
        # For vMF, the bandwidth is directly the concentration parameter.
        if isinstance(bandwidth, torch.Tensor):
            assert not bandwidth.requires_grad, (
                "The bandwidth for the von Mises-Fisher kernel must not require gradients."
            )
            bandwidth = bandwidth.item()  # input to iv function cannot handle tensors
        assert (
            isinstance(bandwidth, float)
            or isinstance(bandwidth, torch.Tensor)
            and bandwidth.dim() == 0
        ), "The bandwidth for the von Mises-Fisher kernel must be a scalar."
        self._bandwidth = bandwidth
        self.inv_bandwidth = bandwidth

    def __call__(self, x1, x2):
        super().__call__(x1, x2)
        x_all = torch.cat([x1, x2], dim=1)
        assert torch.allclose(x_all.norm(dim=-1), torch.ones_like(x_all[..., 0]), atol=1e-5), (
            "The von Mises-Fisher kernel assumes all data to lie on the unit sphere. Please normalize data."
        )

        return torch.exp(self._bandwidth * (x1 * x2).sum(dim=-1))

    def _compute_norm_constant(self, dim):
        assert not check_if_mat(self._bandwidth), (
            "The von Mises-Fisher kernel only support scalar bandwidth arguments."
        )
        # normalizing constant for the vMF kernel. Reference, e.g.: https://www.jmlr.org/papers/volume6/banerjee05a/banerjee05a.pdf
        return (self._bandwidth ** (dim / 2 - 1)) / (
            (2 * math.pi) ** (dim / 2) * float(iv(dim / 2 - 1, self._bandwidth))
        )
