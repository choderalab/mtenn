As of v0.4.0 the `Combination` class has been reworked to be able to run on normal sized
GPUs. Due to the size of the all-atom protein-ligand complex representation, storing all
of the autograd computation graphs for every pose used all the GPU memory. By splitting
the gradient math up into a function of the gradient from each pose, we can reduce the
need to store more than one comp graph at a time. This document contains the derivation
of the split up math.

# `MeanCombination`
Just take the mean of all preds, so the gradient is straightforward:
```math
\Delta G(\theta) = \frac{1}{N} \sum_{n=1}^{N} \Delta G_n (\theta)
\frac{\partial \Delta G(\theta)}{\partial \theta} = \frac{1}{N} \sum_{n=1}^{N} \frac{\partial \Delta G_n (\theta)}{\partial \theta}
```
