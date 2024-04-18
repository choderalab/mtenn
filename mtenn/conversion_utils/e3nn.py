"""
``Representation`` and ``Strategy`` implementations for an E(3)-equivariant model using
`e3nn <https://docs.e3nn.org/en/latest/index.html>`_. The underlying model that we use
is specifically the `January 2021 Network
<https://docs.e3nn.org/en/latest/api/nn/models/gate_points_2101.html>`_ model.
"""
from copy import deepcopy
import torch
from e3nn.nn.models.gate_points_2101 import Network

from mtenn.model import GroupedModel, Model
from mtenn.strategy import ComplexOnlyStrategy, ConcatStrategy, DeltaStrategy


class E3NN(Network):
    """
    ``mtenn`` wrapper around the e3nn model. This class handles construction of the
    model and the formatting into ``Representation`` and ``Strategy`` blocks.
    """

    def __init__(self, *args, model=None, **kwargs):
        """
        Initialize the underlying ``e3nn.nn.models.gate_points_2101.Network`` model. If
        a value is passed for ``model``, builds a new
        ``e3nn.nn.models.gate_points_2101.Network`` model based on those
        hyperparameters, and copies over the weights. Otherwise, all ``*args`` and
        ``**kwargs`` are passed directly to the
        ``e3nn.nn.models.gate_points_2101.Network`` constructor.

        Parameters
        ----------
        model : ``e3nn.nn.models.gate_points_2101.Network``, optional
            e3nn model to use to construct the underlying model
        """
        # If no model is passed, construct e3nn model with model_kwargs,
        #  otherwise copy all parameters and weights over
        if model is None:
            super(E3NN, self).__init__(*args, **kwargs)
            self.model_parameters = kwargs
        else:
            model_kwargs = {
                "irreps_in": model.irreps_in,
                "irreps_hidden": model.irreps_hidden,
                "irreps_out": model.irreps_out,
                "irreps_node_attr": model.irreps_node_attr,
                "irreps_edge_attr": model.irreps_edge_attr,
                "layers": len(model.layers) - 1,
                "max_radius": model.max_radius,
                "number_of_basis": model.number_of_basis,
                "num_nodes": model.num_nodes,
                "reduce_output": model.reduce_output,
            }
            # These need a bit of work to get
            # Use last layer bc guaranteed to be present and is just a Convolution (no
            #  Gate)
            conv = model.layers[-1]
            model_kwargs["radial_layers"] = len(conv.fc.hs) - 2
            model_kwargs["radial_neurons"] = conv.fc.hs[1]
            model_kwargs["num_neighbors"] = conv.num_neighbors
            super(E3NN, self).__init__(**model_kwargs)
            self.model_parameters = model_kwargs
            self.load_state_dict(model.state_dict())

    def forward(self, data):
        """
        Make a prediction of the target property based on an input structure.

        Parameters
        ----------
        data : dict
            This dictionary should at minimum contain entries for:

            * ``"pos"``: Atom coordinates

            * ``"x"``: One-hot encoding of atomic numbers

            And optionally ``"z"``, which stores the node attributes.

        Returns
        -------
        torch.Tensor
            Model prediction
        """
        x = super(E3NN, self).forward(data)
        copy = deepcopy(data)
        copy["x"] = torch.clone(x)
        return copy

    def _get_representation(self, reduce_output=False):
        """
        Copy model and remove last layer.

        Parameters
        ----------
        reduce_output: bool, default=False
            Whether to reduce output across nodes. This should be set to ``True``
            if you want a uniform size tensor for every input size (eg when
            using a ``ConcatStrategy``)

        Returns
        -------
        mtenn.conversion_utils.e3nn.E3NN
            Copied e3nn model with the last layer removed
        """

        # Copy model so initial model isn't affected
        model_copy = deepcopy(self)
        # Remove last layer
        model_copy.layers = model_copy.layers[:-1]
        model_copy.reduce_output = reduce_output

        return model_copy

    def _get_energy_func(self):
        """
        Return copy of last layer of the model.

        Returns
        -------
        e3nn.nn.models.gate_points_2101.Network
            Copy of last layer
        """

        final_conv = deepcopy(self)

        conv_kwargs = deepcopy(self.model_parameters)
        conv_kwargs["layers"] = 0

        new_model = Network(**conv_kwargs)

        new_model.layers[0] = final_conv.layers[-1]

        return new_model

    def _get_delta_strategy(self):
        """
        Build a :py:class:`DeltaStrategy <mtenn.strategy.DeltaStrategy>` object based on
        the calling model.

        Returns
        -------
        mtenn.strategy.DeltaStrategy
            ``DeltaStrategy`` built from the model
        """

        return DeltaStrategy(self._get_energy_func())

    def _get_complex_only_strategy(self):
        """
        Build a :py:class:`ComplexOnlyStrategy <mtenn.strategy.ComplexOnlyStrategy>`
        object based on the calling model.

        Returns
        -------
        mtenn.strategy.ComplexOnlyStrategy
            ``ComplexOnlyStrategy`` built from the model
        """

        return ComplexOnlyStrategy(self._get_energy_func())

    def _get_concat_strategy(self):
        """
        Build a :py:class:`ConcatStrategy <mtenn.strategy.ConcatStrategy>` object using
        the key ``"x"`` to extract the tensor representation from the data dict.

        Returns
        -------
        ConcatStrategy
            ``ConcatStrategy`` for the model
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
        Exposed function to build a :py:class:`Model <mtenn.model.Model>` or
        :py:class:`GroupedModel <mtenn.model.GroupedModel>` from an :py:class:`E3NN
        <mtenn.conversion_utils.e3nn.E3NN>` (or args/kwargs). If no ``model`` is given,
        use the ``model_kwargs``.

        Parameters
        ----------
        model: mtenn.conversion_utils.e3nn.E3NN, optional
            ``E3NN`` model to use to build the ``Model`` object. If not given, use the
            passed ``model_kwargs``
        model_kwargs: dict, optional
            Dictionary used to initialize ``E3NN`` model if nothing is passed for
            ``model``
        grouped: bool, default=False
            Build a ``GroupedModel``
        fix_device: bool, default=False
            If True, make sure the input is on the same device as the model,
            copying over as necessary
        strategy: str, default='delta'
            ``Strategy`` to use to combine representations of the different parts.
            Options are [``delta``, ``concat``, ``complex``]
        combination: mtenn.combination.Combination, optional
            ``Combination`` object to use to combine multiple predictions. A value must
            be passed if ``grouped`` is ``True``
        pred_readout : mtenn.readout.Readout, optional
            ``Readout`` object for the individual energy predictions. If a
            ``GroupedModel`` is being built, this ``Readout`` will be applied to each
            individual prediction before the values are passed to the ``Combination``.
            If a ``Model`` is being built, this will be applied to the single prediction
            before it is returned
        comb_readout : mtenn.readout.Readout, optional
            Readout object for the combined multi-pose prediction, in the case that a
            ``GroupedModel`` is being built. Otherwise, this is ignored

        Returns
        -------
        mtenn.model.Model
            ``Model`` or ``GroupedModel`` containing the desired ``Representation``,
            ``Strategy``, and ``Combination`` and ``Readout`` s as desired
        """

        if model is None:
            model = E3NN(model_kwargs)

        # Construct strategy module based on model and
        #  representation (if necessary)
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

        # Get representation module
        representation = model._get_representation(reduce_output=reduce_output)

        # Check on `combination`
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
