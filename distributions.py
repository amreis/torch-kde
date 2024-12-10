"""Some example probability distributions for testing."""

from abc import ABC, abstractmethod

import torch
from scipy.interpolate import interp1d
from torch.distributions import Normal

NEG_LIM = -5.0
POS_LIM = 5.0
DX = 0.01


class Distribution(ABC):
    @abstractmethod
    def prob(self, x):
        """Probability density."""
        pass

    @abstractmethod
    def sample(self, num_samples):
        """Sample from the distribution."""
        pass


class BartSimpsonDistribution(Distribution):
    """Bart Simpson distribution, see https://www.stat.cmu.edu/~larry/=sml/densityestimation.pdf."""
    def __init__(self):
        super(BartSimpsonDistribution, self).__init__()
        self.inv_cdf = self.compute_inv_cdf()
    
    def prob(self, x):
        prob = (1/2) * Normal(0, 1).log_prob(x).exp()
        for i in range(0, 5):
            prob += (1/10) * Normal((i/2) - 1, 1/10).log_prob(x).exp()
        return prob
    
    def sample(self, num_samples):
        """Uniform samples from the distribution."""
        samples = torch.rand(num_samples)
        # obtain samples from the inverse cdf
        return self.inv_cdf(samples).unsqueeze(1)
    
    def compute_inv_cdf(self):
        """Cumulative distribution function."""
        rg = torch.arange(NEG_LIM, POS_LIM, DX)
        cdf = torch.zeros(rg.size())
        for i in range(rg.size(0)):
            cdf[i] = self.prob(rg[i]) + cdf[i-1]
        cdf = cdf / cdf[-1]
        inv_cdf = invert_function(cdf, rg)
        return inv_cdf

def invert_function(func_tensor, rg):
    # Define the interpolation function
    interp_func = interp1d(func_tensor.numpy(), rg.numpy(), kind='linear')

    # Define the inverse function
    def inverse_func(y):
        # Ensure y is within the range of the function
        y_clipped = torch.clamp(y, func_tensor[0], func_tensor[-1])
        # Interpolate to find the corresponding x value
        return torch.tensor(interp_func(y_clipped.cpu().numpy()))

    return inverse_func
