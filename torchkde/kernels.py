from abc import ABC, abstractmethod
import math

import torch

SUPPORTED_KERNELS = [
    "gaussian"
]

class Kernel(ABC):
    @abstractmethod
    def __call__(self, x):
        pass


class GaussianKernel(Kernel):
    def __init__(self, bandwidth):
        self.bandwidth = bandwidth

    def __call__(self, x):
        return torch.exp(-((x / self.bandwidth)**2).sum(-1)) / \
                ((2 * math.pi)**(0.5))
    