# TorchKDE :fire:

An open source implementation of differentiable kernel density estimation in PyTorch by Klaus-Rudolf Kladny. 

$$\hat{p}(x) = \frac{1}{hn} \sum_{i=1}^n K \left( \frac{x - x_i}{h} \right)$$

## What's included?

### Kernel Density Estimation

KernelDensity supports the same operations as the [KernelDensity class in scikit-learn](https://scikit-learn.org/dev/modules/generated/sklearn.neighbors.KernelDensity.html), but implemented in PyTorch:

```python
from torchkde import KernelDensity
```

Check out `demo_kde.ipynb` for a simple demo.


## (Currently) Supported Settings

The current implementation only provides basic functionality.

**Supported kernels**: Gaussian.

**Supported tree algorithms**: Standard (which corresponds to a simple root tree that returns the entire data set).

**Supported bandwidths**: Only floats, bandwiths estimators such as scott or silverman and are not supported.

## Got an Extension? Create a Pull Request!

In case you do not know how to do that, here are the necessary steps:

1. Fork the repo
2. Create your feature branch (`git checkout -b cool_tree_algorithm`)
3. Commit your changes (`git commit -am 'Add cool tree algorithm'`)
4. Push to the branch (`git push origin cool_tree_algorithm`)
5. Open a Pull Request

## Issues?

If you discover a bug or do not understand something, please let me know at *kkladny [at] tuebingen [dot] mpg [dot] de* and I will fix it!