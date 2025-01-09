# TorchKDE :fire:

![Python Version](https://img.shields.io/badge/python-3.11.11%2B-blue.svg)
![PyTorch Version](https://img.shields.io/badge/pytorch-2.5.1-brightgreen.svg)

A differentiable implementation of [kernel density estimation](https://en.wikipedia.org/wiki/Kernel_density_estimation) in PyTorch by Klaus-Rudolf Kladny.

$$\hat{f}(x) = \frac{1}{h^dn} \sum_{i=1}^n K \left( \frac{x - x_i}{h} \right)$$

## Installation Instructions

Clone the repository, cd into the root directory and run

```bash
pip install .
```

Now you are ready to go!

## What's included?

### Kernel Density Estimation

The `KernelDensity` class supports the same operations as the [KernelDensity class in scikit-learn](https://scikit-learn.org/dev/modules/generated/sklearn.neighbors.KernelDensity.html), but implemented in PyTorch and differentiable with respect to input data. Here is a little taste:

```python
from torchkde import KernelDensity
import torch

multivariate_normal = torch.distributions.MultivariateNormal(torch.ones(2), torch.eye(2))
X = multivariate_normal.sample((1000,)) # create data
X.requires_grad = True # make differentiable
kde = KernelDensity(bandwidth=1.0, kernel='gaussian') # create kde object
_ = kde.fit(X) # fit kde to data

X_new = multivariate_normal.sample((100,)) # create new data 
logprob = kde.score_samples(X_new)

logprob.grad_fn # is not None
```

You may also check out `demo_kde.ipynb` for a simple demo on the [Bart Simpson distribution](https://www.stat.cmu.edu/~larry/=sml/densityestimation.pdf), which yields the following density estimate:

<p align="center">
<img src="/plots/bart_simpson_kde.svg" width="500">
</p>

## (Currently) Supported Settings

The current implementation only provides basic functionality.

**Supported kernels**: Gaussian, Epanechnikov (sampling new data points is only supported for Gaussian).

**Supported tree algorithms**: Standard (which corresponds to a simple root tree that returns the entire data set).

**Supported bandwidths**: Only floats, bandwiths estimators such as scott or silverman and are not supported.

## Got an Extension? Create a Pull Request!

In case you do not know how to do that, here are the necessary steps:

1. Fork the repo
2. Create your feature branch (`git checkout -b cool_tree_algorithm`)
3. Run the unit tests (`python -m tests.test_kde`) and only proceed if the script outputs "OK".
4. Commit your changes (`git commit -am 'Add cool tree algorithm'`)
5. Push to the branch (`git push origin cool_tree_algorithm`)
6. Open a Pull Request

## Issues?

If you discover a bug or do not understand something, please let me know at *kkladny [at] tuebingen [dot] mpg [dot] de* and I will fix it!
