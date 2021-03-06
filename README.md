MTENN
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/mtenn/workflows/CI/badge.svg)](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/mtenn/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/MTENN/branch/master/graph/badge.svg)](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/MTENN/branch/master)


Modular Training and Evaluation of Neural Networks

### Copyright

Copyright (c) 2022, Benjamin Kaminow

### Minimal usage example
A minimal example of how to use this package (with some random data).

First generate the data
```python
import torch

## Complex data
##  z: random integers (0-10)
##  pos: random position floats
z_comp = torch.randint(10, (10,))
pos_comp = torch.rand((10,3))

## Protein data
z_prot = z_comp[:7]
pos_prot = pos_comp[:7,:]

## Ligand data
z_lig = z_comp[7:]
pos_lig = pos_comp[7:,:]
```

Construct the ```mtenn``` SchNet models
```python
from mtenn.conversion_utils import SchNet

## Generate an instance of the mtenn SchNet model
m = SchNet()

## Use that model to construct a Model object using the delta strategy and one
##  using the concat strategy
delta_model = SchNet.get_model(model=m, strategy='delta')
concat_model = SchNet.get_model(model=m, strategy='concat')
```

Rearrange data to pass to ```mtenn```
```python
## Our SchNet models take a tuple of (atomic_numbers, positions)

## Complex representation
rep_comp = (z_comp, pos_comp)

## Protein representation
rep_prot = (z_prot, pos_prot)

## Ligand representation
rep_lig = (z_lig, pos_lig)
```

Calculate energies using the different models
```python
## First predict energies using the vanilla SchNet model
e_comp = m(rep_comp)
e_prot = m(rep_prot)
e_lig = m(rep_lig)
## Calculate delta energy using
delta_e_og = e_comp - (e_prot + e_lig)

## Use the mtenn Model object to directly predict the same delta energy with
delta_e_new = delta_model(rep_comp, rep_prot, rep_lig)
# won't be exactly equal bc floating point inaccuracy
assert torch.isclose(delta_e_og, delta_e_new)

## Use the concat Model to predict delta energy (this will be different from
##  the other predicted energies)
concat_e = concat_model(rep_comp, rep_prot, rep_lig)

print(f'Using vanilla SchNet model: {delta_e_og.item():0.5f}')
print(f'Using delta Model: {delta_e_new.item():0.5f}')
print(f'Using concat Model: {concat_e.item():0.5f}')
```

#### Acknowledgements

Project based on the
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.6.
