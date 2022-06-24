"""
Representation and strategy for SchNet model.
"""
from copy import deepcopy
import torch
from torch_geometric.nn.models import SchNet as PygSchNet

from ..model import ConcatStrategy, DeltaStrategy, Model

class SchNet(PygSchNet):
    def __init__(self, model=None):
        ## If no model is passed, construct default SchNet model, otherwise copy
        ##  all parameters and weights over
        if model is None:
            super(SchNet, self).__init__()
        else:
            atomref = model.atomref.weight.detach().clone()
            model_params = (model.hidden_channels, model.num_filters,
                model.num_interactions, model.num_gaussians,
                model.cutoff, model.max_num_neighbors,model.readout,
                model.dipole, model.mean, model.std, atomref)
            super(SchNet, self).__init__(*model_params)
            self.load_state_dict(model.state_dict())

    def forward(self, data):
        return(super(SchNet, self).forward(data[0], data[1]))

def get_representation(model):
    """
    Input model, remove last layer.

    Parameters
    ----------
    model: SchNet
        SchNet model

    Returns
    -------
    SchNet
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
    model: SchNet
        SchNet model

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
    model: SchNet
        SchNet model

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
    model: SchNet
        SchNet model

    Returns
    -------
    ConcatStrategy
        ConcatStrategy built from `model`
    """
    pass

def get_model(model, strategy: str='delta'):
    ## First get representation module
    representation = get_representation(model)

    ## Construct strategy module based on model and
    ##  representation (if necessary)
    if strategy == 'delta':
        strategy = get_delta_strategy(model)
    elif strategy == 'concat':
        strategy = ConcatStrategy()

    return(Model(representation, strategy))
