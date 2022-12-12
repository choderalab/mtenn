"""
Representation and strategy for SchNet model.
"""
from copy import deepcopy
import torch
from torch_geometric.nn.models import SchNet as PygSchNet

from ..model import (
    BoltzmannCombination,
    ConcatStrategy,
    DeltaStrategy,
    GroupedModel,
    MeanCombination,
    Model,
    PIC50Readout,
)


class SchNet(PygSchNet):
    def __init__(self, model=None):
        ## If no model is passed, construct default SchNet model, otherwise copy
        ##  all parameters and weights over
        if model is None:
            super(SchNet, self).__init__()
        else:
            try:
                atomref = model.atomref.weight.detach().clone()
            except AttributeError:
                atomref = None
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
    def get_model(
        model=None,
        grouped=False,
        fix_device=False,
        strategy: str = "delta",
        combination=None,
        pred_readout=None,
        comb_readout=None,
    ):
        """
        Exposed function to build a Model object from a SchNet object. If none
        is provided, a default model is initialized.

        Parameters
        ----------
        model: SchNet, optional
            SchNet model to use to build the Model object. If left as none, a
            default model will be initialized and used
        grouped: bool, default=False
            Whether this model should accept groups of inputs or one input at a
            time.
        fix_device: bool, default=False
            If True, make sure the input is on the same device as the model,
            copying over as necessary.
        strategy: str, default='delta'
            Strategy to use to combine representation of the different parts.
            Options are ['delta', 'concat']
        combination: Combination, optional
            Combination object to use to combine predictions in a group. A value
            must be passed if `grouped` is `True`.
        pred_readout : Readout
            Readout object for the energy predictions. If `grouped` is `False`,
            this option will still be used in the construction of the `Model`
            object.
        comb_readout : Readout
            Readout object for the combination output.

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
        strategy = strategy.lower()
        if strategy == "delta":
            strategy = model._get_delta_strategy()
        elif strategy == "concat":
            strategy = ConcatStrategy()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        ## Check on `combination`
        if grouped and (combination is None):
            raise ValueError(
                f"Must pass a value for `combination` if `grouped` is `True`."
            )

        if grouped:
            return GroupedModel(
                representation,
                strategy,
                combination,
                pred_readout,
                comb_readout,
                fix_device,
            )
        else:
            return Model(representation, strategy, pred_readout, fix_device)
