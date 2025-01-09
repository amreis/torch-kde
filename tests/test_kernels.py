"""Test that checks whether the kernel density estimator integrates to 1 for various settings."""

from itertools import product
import unittest

import torch

from torchkde.kernels import *
from torchkde.modules import KernelDensity

BANDWIDTHS = [0.5, 1.0, 10.0]
DIMS = [1, 2]
TOLERANCE = 5e-2

N = 10000
GRID_N = 100
GRID_RG = 100


class KernelTestCase(unittest.TestCase):
    def test_kernels(self):
        for kernel_str, bandwidth, dim in product(SUPPORTED_KERNELS, BANDWIDTHS, DIMS):
            # sample data from a normal distribution
            mean = torch.ones(dim)
            covariance_matrix = torch.eye(dim) 

            # Create the multivariate Gaussian distribution
            multivariate_normal = torch.distributions.MultivariateNormal(mean, covariance_matrix)
            X = multivariate_normal.sample((N,))
            # Fit a kernel density estimator to the data
            kde = KernelDensity(bandwidth=bandwidth, kernel=kernel_str)
            kde.fit(X)
            # assess whether the kernel integrates to 1
            # evaluate the kernel density estimator at a grid of 2D points
            # Create ranges for each dimension
            ranges = [torch.linspace(-GRID_RG, GRID_RG, GRID_N) for _ in range(dim)]
            # Create the d-dimensional meshgrid
            meshgrid = torch.meshgrid(*ranges, indexing='ij')  # 'ij' indexing for Cartesian coordinates

            # Convert meshgrid to a single tensor of shape (n_points, d)
            grid_points = torch.stack(meshgrid, dim=-1).reshape(-1, dim)
            probs = kde.score_samples(grid_points).exp()
            delta = (GRID_RG * 2) / GRID_N
            integral = probs.sum() * (delta**dim)
            self.assertTrue((integral - 1.0).abs() < TOLERANCE, 
                            f"""Kernel {kernel_str}, for dimensionality {str(dim)} 
                            and bandwidth {str(bandwidth)} does not integrate to 1.""")


if __name__ == "__main__":
    torch.manual_seed(0) # ensure reproducibility
    unittest.main()
