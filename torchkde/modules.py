import torch
from torch import nn

from .utils import ensure_two_dimensional, check_if_mat
from .algorithms import RootTree, SUPPORTED_ALGORITHMS
from .bandwidths import SUPPORTED_BANDWIDTHS, compute_bandwidth
from .kernels import (GaussianKernel, 
                      EpanechnikovKernel, 
                      ExponentialKernel, 
                      TopHatKernel, 
                      SUPPORTED_KERNELS)


ALG_DICT = {
    "standard": RootTree
    }


KERNEL_DICT = {
    "gaussian": GaussianKernel,
    "epanechnikov": EpanechnikovKernel,
    "exponential": ExponentialKernel,
    "tophat-approx": TopHatKernel
}


class KernelDensity(nn.Module):
    """Analag to the KernelDensity class in sklearn.neighbors 
    (see https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/neighbors/_kde.py)."""

    def __init__(
        self,
        *,
        bandwidth=1.0,
        algorithm="standard",
        kernel="gaussian",
        kernel_kwargs=None
    ):
        if not isinstance(bandwidth, str):
            assert bandwidth > 0, "Bandwidth must be positive."
            self.bandwidth = bandwidth**2 # square the bandwidth to match sklearn's implementation
        else:
            self.bandwidth = bandwidth
        assert kernel in SUPPORTED_KERNELS, f"Kernel {kernel} not supported."
        self.kernel = kernel
        if kernel_kwargs is None:
            kernel_kwargs = {}
        self.kernel_module = KERNEL_DICT[kernel](**kernel_kwargs)
        assert algorithm in SUPPORTED_ALGORITHMS, f"Algorithm {algorithm} not supported."
        self.algorithm = algorithm
        self.is_fitted = False
        self.n_samples = None
        self.n_features = None
        self.data = None

        if algorithm not in SUPPORTED_ALGORITHMS:
            raise ValueError(f"Algorithm {algorithm} not supported")
        
        if kernel not in SUPPORTED_KERNELS:
            raise ValueError(f"Kernel {kernel} not supported")

        if not isinstance(bandwidth, (float, torch.Tensor)) and bandwidth not in SUPPORTED_BANDWIDTHS:
            raise ValueError(f"Bandwidth {bandwidth} not supported")


    def fit(self, X, sample_weight=None):
        """Fit the Kernel Density model on the data.

        Parameters
        ----------
        X : torch tensor of shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.
        sample_weight : torch tensor of shape (n_samples,), default=None
            Weights for each sample.  If None, all samples are assigned
            equal weight.

        Returns
        -------
        self : object
            Returns the instance itself.
        """

        self.tree_ = ALG_DICT[self.algorithm]()
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        self.tree_.build(X)
        self.bandwidth = compute_bandwidth(X, self.bandwidth)
        self.kernel_module.bandwidth = self.bandwidth
        if sample_weight is None:
            self.sample_weight = torch.ones(self.n_samples)
        else:
            assert sample_weight.shape[0] == self.n_samples, "Sample weights must have the same length as the data."
            assert sample_weight.dim() == 1, "Sample weights must be one-dimensional."
            assert sample_weight.min() >= 0, "Sample weights must be non-negative."
            self.sample_weight = sample_weight
        self.is_fitted = True

        return self

    def score_samples(self, X):
        """Compute the log-likelihood of each sample under the model.

        Parameters
        ----------
        X : torch Tensor of shape (n_samples, n_features)
            An array of points to query.  Last dimension should match dimension
            of training data (n_features).
        Returns
        -------
        density : torch Tensor of shape (n_samples,)
            Log-likelihood of each sample in `X`. These are normalized to be
            probability densities, so values will be low for high-dimensional
            data.
        """
        assert self.is_fitted, "Model must be fitted before scoring samples."
   
        X_neighbors = self.tree_.query(X, return_distance=False)
        # Compute log-density estimation with a kernel function
        log_density = []
        # Compute normalization part from bandwidth matrix
        bw_norm = torch.sqrt(torch.det(self.bandwidth)) if check_if_mat(self.bandwidth) else self.bandwidth**(self.n_features/2)
        # looping to avoid memory issues
        for i, x in enumerate(X):
            # Compute pairwise differences between the current point and neighbors
            differences = x - X_neighbors[i]
            # Apply the kernel function to each difference
            kernel_values = self.kernel_module(differences)
            # Sum kernel values and normalize
            density = (self.sample_weight * kernel_values).sum(-1) / (bw_norm * self.sample_weight.sum())
            # Compute the log-density
            log_density.append(density.log())

        # Convert the list of log-density values into a tensor
        log_density = torch.stack(log_density, dim=0)
        return log_density


    def score(self, X):
        """Compute the total log-likelihood under the model.

        Parameters
        ----------
        X : torch Tensor of shape (n_samples, n_features)
            An array of points to query.  Last dimension should match dimension
            of training data (n_features).

        Returns
        -------
        logprob : float
            Total log-likelihood of the data in X. This is normalized to be a
            probability density, so the value will be low for high-dimensional
            data.
        """
        return self.score_samples(X).sum()

    def sample(self, n_samples=1):
        """Generate random samples from the model.

        Parameters
        ----------
        n_samples : int, default=1
            Number of samples to generate.

        Returns
        -------
        X : array-like of shape (n_samples, n_features)
            List of samples.
        """
        assert self.is_fitted, "Model must be fit before sampling."

        if self.kernel not in ["gaussian"]:
            raise NotImplementedError()

        data = torch.tensor(self.tree_.data)
        idxs = torch.randint(data.shape[0], (n_samples,))

        X = self.bandwidth * torch.randn(n_samples, data.shape[1]) + data[idxs]

        return ensure_two_dimensional(X)
    