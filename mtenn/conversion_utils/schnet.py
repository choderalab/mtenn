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
            model_params = (
                model.hidden_channels,
                model.num_filters,
                model.num_interactions,
                model.num_gaussians,
                model.cutoff,
                model.max_num_neighbors,
                model.readout,
                model.dipole,
                model.mean,
                model.std,
                atomref,
            )
            super(SchNet, self).__init__(*model_params)
            self.load_state_dict(model.state_dict())

    def forward(self, data):
        return super(SchNet, self).forward(data["z"], data["pos"])

    def _get_representation(self):
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
        model_copy = deepcopy(self)
        ## Replace final linear layer with an identity module
        model_copy.lin2 = torch.nn.Identity()

        return model_copy

    def _get_energy_func(self):
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

        return deepcopy(self.lin2)

    def _get_delta_strategy(self):
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

        return DeltaStrategy(self._get_energy_func())

    @staticmethod
    def get_model(model=None, strategy: str = "delta"):
        """
        Exposed function to build a Model object from a SchNet object. If none
        is provided, a default model is initialized.

        Parameters
        ----------
        model: SchNet, optional
            SchNet model to use to build the Model object. If left as none, a
            default model will be initialized and used
        strategy: str, default='delta'
            Strategy to use to combine representation of the different parts.
            Options are ['delta', 'concat']

        Returns
        -------
        Model
            Model object containing the desired Representation and Strategy
        """
        if model is None:
            model = SchNet()

        ## First get representation module
        representation = model._get_representation()

        ## Construct strategy module based on model and
        ##  representation (if necessary)
        if strategy == "delta":
            strategy = model._get_delta_strategy()
        elif strategy == "concat":
            strategy = ConcatStrategy()

        return Model(representation, strategy)
