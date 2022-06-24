"""
Representation and strategy for SchNet model.
"""
from copy import deepcopy
import torch

from ..model import ConcatStrategy, DeltaStrategy, Model

def get_reprentation(model):
    """
    Input model, remove last layer.

    Parameters
    ----------
    model: torch_geometric.nn.models.SchNet
        pyg SchNet model

    Returns
    -------
    torch_geometric.nn.models.SchNet
        Copied SchNet model with the last layer replaced by an Identity module
    """

    ## Copy model so initial model isn't affected
    model_copy = deepcopy(model)
    ## Replace final linear layer with an identity module
    model_copy.lin2 = torch.nn.Identity()

    return(model_copy)

def get_energy_func(model):
    """
    Return last layer of the model.

    Parameters
    ----------
    model: torch_geometric.nn.models.SchNet
        pyg SchNet model

    Returns
    -------
    torch.nn.modules.linear.Linear
        Copy of `model`'s last layer
    """

    return(deepcopy(model.lin2))

def get_delta_strategy(model):
    """
    Build a DeltaStrategy object based on the passed model.

    Parameters
    ----------
    model: torch_geometric.nn.models.SchNet
        pyg SchNet model

    Returns
    -------
    DeltaStrategy
        DeltaStrategy built from `model`
    """

    return(DeltaStrategy(get_energy_func(model)))

def get_concat_strategy(model):
    """
    Build a ConcatStrategy object based on the passed model.

    Parameters
    ----------
    model: torch_geometric.nn.models.SchNet
        pyg SchNet model

    Returns
    -------
    ConcatStrategy
        ConcatStrategy built from `model`
    """
    pass

def get_model(model, strategy: str='delta'):
    ## First get representation module
    representation = get_reprentation(model)

    ## Construct strategy module based on model and
    ##  representation (if necessary)
    if strategy == 'delta':
        strategy = get_delta_strategy(model)
    elif strategy == 'concat':
        strategy = ConcatStrategy()

    return(Model(representation, strategy))
