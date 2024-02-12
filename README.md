MTENN
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/choderalab/mtenn/workflows/CI/badge.svg)](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/mtenn/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/choderalab/mtenn/branch/main/graph/badge.svg)](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/MTENN/branch/main)
[![Documentation Status](https://readthedocs.org/projects/mtenn/badge/?version=latest)](https://mtenn.readthedocs.io/en/latest/?badge=latest)

Modular Training and Evaluation of Neural Networks

### Copyright

Copyright (c) 2022, Benjamin Kaminow

### Minimal usage example
Building models should be done using the `mtenn.config` API.
A small example for a SchNet model is shown below, but more details for SchNet and other models can be found in the respective class definitions.

We will construct a SchNet model with default parameters and a delta G strategy for combining our complex, protein, and ligand representations.
We will leave our predictions in the returned implicit kT units (ie no Readout block).
```python
from mtenn.config import SchNetModelConfig

# Create the config using all default parameters (which includes the delta G strategy)
model_config = SchNetModelConfig()

# Build the actual pytorch model
model = model.build()
```

The input passed to this model should be a `dict` with the following keys (based on the underlying model):
* `SchNet`
    * `z`: Tensor of atomic number for each atom, shape of `(n,)`
    * `pos`: Tensor of coordinates for each atom, shape of `(n,3)`
* `E3NN`
    * `x`: Tensor of one-hot encodings of element for each atom, shape of `(n,one_hot_length)`
    * `pos`: Tensor of coordinates for each atom, shape of `(n,3)`
    * `z`: Tensor of bool labels of whether each atom is a protein atom (`False`) or ligand atom (`True`), shape of `(n,)`
* `GAT`
    * `g`: DGL graph object

The prediction can then be generated simply with:
```python
import torch

# Using random data just for demonstration purposes
pose = {"z": torch.randint(low=1, high=17, size=(100,)), "pos": torch.rand((100, 3))}
pred = model(pose)
```

### Installation

`mtenn` is now on `conda-forge`! To install, simply run
```bash
mamba install -c conda-forge mtenn
```


#### Acknowledgements

Project based on the
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.6.
