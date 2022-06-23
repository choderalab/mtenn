MTENN
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/mtenn/workflows/CI/badge.svg)](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/mtenn/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/MTENN/branch/master/graph/badge.svg)](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/MTENN/branch/master)


Modular Training and Evaluation of Neural Networks

### Copyright

Copyright (c) 2022, Benjamin Kaminow

### Minimal usage example
A minimal example of how to use this package (based on [the pytorch Quickstart
guide](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)).

First load the data
```python
from torchvision import datasets
from torchvision.transforms import ToTensor

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)
```

Construct the ```pytorch``` model, loss function, and optimizer
```python
import torch
from torch import nn

# Just use cpu for now
device = 'cpu'

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
```

Build the ```mtenn``` systems
```python
from mtenn.training import Trainer
from mtenn.model import Model

m = Model(model, lambda model, d: model(d))
t = Trainer(loss_fn, optimizer, m)
```

Rearrange data to pass to ```mtenn```
```python
train_Xs, train_ys = zip(*[(X, y) for X, y in training_data])
train_datasets = [train_Xs]
train_targets = [train_ys]

eval_Xs, eval_ys = zip(*[(X, y) for X, y in training_data])
eval_datasets = [eval_Xs]
eval_targets = [eval_ys]
```

Train the model for 10 epochs
```python
train_loss, eval_loss = t.train(train_datasets, eval_datasets, train_targets,
    eval_targets, 10)
print(train_loss[0].mean())
print(eval_loss[0].mean())
```

#### Acknowledgements

Project based on the
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.6.
