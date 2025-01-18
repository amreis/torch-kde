# TorchKDE :fire:

![Python Version](https://img.shields.io/badge/python-3.11.11%2B-blue.svg)
![PyTorch Version](https://img.shields.io/badge/pytorch-2.5.1-brightgreen.svg)
![Tests](https://github.com/rudolfwilliam/torch-kde/actions/workflows/ci.yml/badge.svg)
[![DOI](https://zenodo.org/badge/901331908.svg)](https://doi.org/10.5281/zenodo.14674657)

A differentiable implementation of [kernel density estimation](https://en.wikipedia.org/wiki/Kernel_density_estimation) in PyTorch by Klaus-Rudolf Kladny.

$$\hat{f}(x) = \frac{1}{|H|^{\frac{1}{2}} n} \sum_{i=1}^n K \left( H^{-\frac{1}{2}} \left( x - x_i \right) \right)$$

## Installation Instructions

Clone the repository, `cd` into the root directory and run

```bash
pip install .
```

Now you are ready to go! If you would also like to run the code in the Jupyter notebooks or contribute to this package, please also install the packages in the `requirements.txt` (inside of an environment):

```bash
pip install -r requirements.txt
```

## What's included?

### Kernel Density Estimation

The `KernelDensity` class supports the same operations as the [KernelDensity class in scikit-learn](https://scikit-learn.org/dev/modules/generated/sklearn.neighbors.KernelDensity.html), but implemented in PyTorch and differentiable with respect to input data. Here is a little taste:

```python
from torchkde import KernelDensity
import torch

multivariate_normal = torch.distributions.MultivariateNormal(torch.ones(2), torch.eye(2))
X = multivariate_normal.sample((1000,)) # create data
X.requires_grad = True # enable differentiation
kde = KernelDensity(bandwidth=1.0, kernel='gaussian') # create kde object with isotropic bandwidth matrix
_ = kde.fit(X) # fit kde to data

X_new = multivariate_normal.sample((100,)) # create new data 
logprob = kde.score_samples(X_new)

logprob.grad_fn # is not None
```

You may also check out `demo_kde.ipynb` for a simple demo on the [Bart Simpson distribution](https://www.stat.cmu.edu/~larry/=sml/densityestimation.pdf), which yields the following density estimate:

<p align="center">
<img src="/plots/bart_simpson_kde.svg" width="500">
</p>

## Supported Settings

The current implementation provides the following functionality:

<div align="center">

| Feature                  | Supported Values            |
|--------------------------|-----------------------------|
| Kernels                  | Gaussian, Epanechnikov      |
| Tree Algorithms          | Standard                    |
| Bandwidths               | Float (Isotropic bandwidth matrix), Scott, Silverman |

</div>

## Got an Extension? Create a Pull Request!

In case you do not know how to do that, here are the necessary steps:

1. Fork the repo
2. Create your feature branch (`git checkout -b cool_tree_algorithm`)
3. Run the unit tests (`python -m tests.test_kde`) and only proceed if the script outputs "OK".
4. Commit your changes (`git commit -am 'Add cool tree algorithm'`)
5. Push to the branch (`git push origin cool_tree_algorithm`)
6. Open a Pull Request

## Issues?

If you discover a bug or do not understand something, please create an issue or let me know directly at *kkladny [at] tuebingen [dot] mpg [dot] de*! I am also happy to take requests for implementing specific functionalities.


<div align="center">

> "In God we trust. All others must bring data."
> 
> — W. Edwards Deming
> 
</div>