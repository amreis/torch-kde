from torch import nn


class KernelDensityRegularizer(nn.Module):
    """Module that computes the KDE of a set of points at a given grid."""
    def __init__(self, grid, bandwidth, kernel="gaussian"):
        """
        Parameters
        ----------
        grid : torch.Tensor
            The grid points at which to evaluate the KDE.
        bandwidth : float
            The bandwidth of the kernel.
        kernel : str
            The kernel to use. Must be one of SUPPORTED_KERNELS.
        """
        super().__init__()
        self.grid = grid
        self.bandwidth = bandwidth
        if kernel == "gaussian":
            #self.kernel = GaussianKernel(bandwidth)
            pass
        else:
            raise ValueError(f"Kernel {kernel} not supported")
    
    def forward(self, x):
        # Evaluate the KDE at the grid points
        return self.kernel((x[:, None] - self.grid[None, :]).sum(-1))*(1/(len(x)*self.bandwidth))
    