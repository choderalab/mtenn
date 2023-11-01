"""
Representation and strategy for e3nn model.
"""
from copy import deepcopy
import torch
from e3nn.nn.models.gate_points_2101 import Network

from mtenn.model import GroupedModel, Model
from mtenn.strategy import ComplexOnlyStrategy, ConcatStrategy, DeltaStrategy


class E3NN(Network):
    def __init__(self, model=None, model_kwargs=None):
        ## If no model is passed, construct E3NN model with model_kwargs,
        ##  otherwise copy all parameters and weights over
        if model is None:
            super(E3NN, self).__init__(**model_kwargs)
            self.model_parameters = model_kwargs
        else:
            # this will need changing to include  model features of e3nn
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
            super(E3NN, self).__init__(*model_params)
            self.model_parameters = model_params
            self.load_state_dict(model.state_dict())

    def forward(self, data):
        x = super(E3NN, self).forward(data)
        copy = deepcopy(data)
        copy["x"] = torch.clone(x)
        return copy

    def _get_representation(self, reduce_output=False):
        """
        Input model, remove last layer.
        Parameters
        ----------
        reduce_output: bool, default=False
            Whether to reduce output across nodes. This should be set to True
            if you want a uniform size tensor for every input size (eg when
            using a ConcatStrategy).

        Returns
        -------
        E3NN
            Copied e3nn model with the last layer removed
        """

        ## Copy model so initial model isn't affected
        model_copy = deepcopy(self)
        ## Remove last layer
        model_copy.layers = model_copy.layers[:-1]
        model_copy.reduce_output = reduce_output

        return model_copy

    def _get_energy_func(self):
        """
        Return last layer of the model.
        Parameters
        ----------
        model: e3nn
            e3nn model
        Returns
        -------
        e3nn.nn.models.gate_points_2101.Network
            Copy of `model`'s last layer
        """

        final_conv = deepcopy(self)

        conv_kwargs = deepcopy(self.model_parameters)
        conv_kwargs["layers"] = 0

        new_model = Network(**conv_kwargs)

        new_model.layers[0] = final_conv.layers[-1]

        return new_model

    def _get_delta_strategy(self):
        """
        Build a DeltaStrategy object based on the calling model.

        Returns
        -------
        DeltaStrategy
            DeltaStrategy built from the model
        """

        return DeltaStrategy(self._get_energy_func())

    def _get_complex_only_strategy(self):
        """
        Build a ComplexOnlyStrategy object based on the passed model.

        Returns
        -------
        ComplexOnlyStrategy
            ComplexOnlyStrategy built from `self`
        """

        return ComplexOnlyStrategy(self._get_energy_func())

    def _get_concat_strategy(self):
        """
        Build a ConcatStrategy object using the key "x" to extract the tensor
        representation from the data dict.

        Returns
        -------
        ConcatStrategy
            ConcatStrategy for the model
        """

        return ConcatStrategy(extract_key="x")

    @staticmethod
    def get_model(
        model=None,
        model_kwargs=None,
        grouped=False,
        fix_device=False,
        strategy: str = "delta",
        combination=None,
        pred_readout=None,
        comb_readout=None,
    ):
        """
        Exposed function to build a Model object from a E3NN object. If none
        is provided, a default model is initialized.

        Parameters
        ----------
        model: E3NN, optional
            E3NN model to use to build the Model object. If left as none, a
            default model will be initialized and used
        model_kwargs: dict, optional
            Dictionary used to initialize E3NN model if model is not passed in
        grouped: bool, default=False
            Whether this model should accept groups of inputs or one input at a
            time.
        fix_device: bool, default=False
            If True, make sure the input is on the same device as the model,
            copying over as necessary.
        strategy: str, default='delta'
            Strategy to use to combine representation of the different parts.
            Options are ['delta', 'concat', 'complex']
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

        #
        if model is None:
            model = E3NN(model_kwargs)

        ## Construct strategy module based on model and
        ##  representation (if necessary)
        strategy = strategy.lower()
        if strategy == "delta":
            strategy = model._get_delta_strategy()
            reduce_output = False
        elif strategy == "concat":
            strategy = model._get_concat_strategy()
            reduce_output = True
        elif strategy == "complex":
            strategy = model._get_complex_only_strategy()
            reduce_output = False
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        ## Get representation module
        representation = model._get_representation(reduce_output=reduce_output)

        ## Check on `combination`
        if grouped and (combination is None):
            raise ValueError(
                "Must pass a value for `combination` if `grouped` is `True`."
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
