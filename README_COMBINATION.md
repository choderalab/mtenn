As of v0.4.0 the `Combination` class has been reworked to be able to run on normal sized
GPUs. Due to the size of the all-atom protein-ligand complex representation, storing all
of the autograd computation graphs for every pose used all the GPU memory. By splitting
the gradient math up into a function of the gradient from each pose, we can reduce the
need to store more than one comp graph at a time. This document contains the derivation
of the split up math.

# `MSE Loss`
```math
L = (\Delta G_{\mathrm{pred}} \left ( \theta \right ) - \Delta G_{\mathrm{target}})^2
```
```math
\frac{\partial L}{\partial \theta} = 2(\Delta G_{\mathrm{pred}} \left ( \theta \right ) - \Delta G_{\mathrm{target}}) \frac{\partial \Delta G_{\mathrm{pred}} \left ( \theta \right )}{\partial \theta}
```

# `MeanCombination`
Just take the mean of all preds, so the gradient is straightforward:
```math
\Delta G(\theta) = \frac{1}{N} \sum_{n=1}^{N} \Delta G_n (\theta)
```
```math
\frac{\partial \Delta G(\theta)}{\partial \theta} = \frac{1}{N} \sum_{n=1}^{N} \frac{\partial \Delta G_n (\theta)}{\partial \theta}
```

# `MaxCombination`
Combine according to a smooth max approximation using LSE:
```math
\Delta G(\theta) = \frac{-1}{t} \mathrm{ln} \sum_{n=1}^N \mathrm{exp} (-t \Delta G_n (\theta))
```
```math
Q = \mathrm{ln} \sum_{n=1}^N \mathrm{exp} (-t \Delta G_n (\theta))
```
```math
\frac{\partial \Delta G(\theta)}{\partial \theta} = \frac{1}{\sum_{n=1}^N \mathrm{exp} (-t \Delta G_n (\theta))} \sum_{n=1}^N \left[ \frac{\partial \Delta G_n (\theta)}{\partial \theta} \mathrm{exp} (-t \Delta G_n (\theta)) \right]
```
```math
\frac{\partial \Delta G(\theta)}{\partial \theta} = \frac{1}{\mathrm{exp}(Q)} \sum_{n=1}^N \left[ \mathrm{exp} \left( -t \Delta G_n (\theta) \right) \frac{\partial \Delta G_n (\theta)}{\partial \theta} \right]
```
```math
\frac{\partial \Delta G(\theta)}{\partial \theta} = \sum_{n=1}^N \left[ \mathrm{exp} \left( -t \Delta G_n (\theta) - Q \right) \frac{\partial \Delta G_n (\theta)}{\partial \theta} \right]
```
# `BoltzmannCombination`
Combine according to Boltzmann weighting:
```math
\Delta G(\theta) = \sum_{n=1}^{N} w_n \Delta G_n (\theta)
```

```math
w_n = \mathrm{exp} \left[ -\Delta G_n (\theta) - \mathrm{ln} \sum_{i=1}^N \mathrm{exp} (-\Delta G_i (\theta)) \right]
```

```math
Q = \mathrm{ln} \sum_{n=1}^N \mathrm{exp} (-\Delta G_n (\theta))
```

```math
\frac{\partial \Delta G(\theta)}{\partial \theta} = \sum_{n=1}^N \left[ \frac{\partial w_n}{\partial \theta} \Delta G_n (\theta) + w_n \frac{\partial \Delta G_n (\theta)}{\partial \theta} \right]
```

```math
\frac{\partial w_n}{\partial \theta} = \mathrm{exp} \left[ -\Delta G_n (\theta) - Q \right] \left[ \frac{-\partial \Delta G_n (\theta)}{\partial \theta} - \frac{\partial Q}{\partial \theta}  \right]
```

```math
\frac{\partial Q}{\partial \theta} = \frac{1}{\sum_{n=1}^N \mathrm{exp} (-\Delta G_n (\theta))} \sum_{i=1}^{N} \left[ \mathrm{exp} (-\Delta G_i (\theta)) \frac{-\partial \Delta G_i (\theta)}{\partial \theta} \right]
```

```math
\frac{\partial Q}{\partial \theta} = \frac{-1}{\mathrm{exp} (Q)} \sum_{n=1}^{N} \left[ \mathrm{exp} (-\Delta G_n (\theta)) \frac{\partial \Delta G_n (\theta)}{\partial \theta} \right]
```

```math
\frac{\partial Q}{\partial \theta} =  -\sum_{n=1}^{N} \left[ \mathrm{exp} (-\Delta G_n (\theta) - Q) \frac{\partial \Delta G_n (\theta)}{\partial \theta} \right]
```
