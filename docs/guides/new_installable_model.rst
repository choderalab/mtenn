.. _new-installable-model-guide:

Adding a new installable model
==============================

There are two main steps to adding a new model to ``mtenn``.
First, we need to write the wrapper class that will live in :py:mod:`mtenn.conversion_utils`.
After that we will define a config class for our new model in :py:mod:`mtenn.config`.
This guide will use the PyTorch Geometric `SchNet <https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.SchNet.html#torch_geometric.nn.models.SchNet>`_ model as an example, but the steps should be more or less the same for any model that can be installed via ``conda``/``pip``.
The steps should be pretty similar for non-packaged models, but may require some additional manipulation of the ``PYTHONPATH`` environment variable.

.. _new-inst-model-conv-utils:

Adding the model wrapper to :py:mod:`mtenn.conversion_utils`
------------------------------------------------------------

Although mostly straightforward, this process is complicated by the fact that it needs to be done in a somewhat ad hoc manner, as the architecture for each model is different.
For use with the overall ``mtenn`` structure, we need to be able to decompose the underlying into two parts: the first that takes in a structure and outputs an n-dimensional vector representation, and the second that takes in an n-dimensional vector representation and outputs a scalar energy prediction.

In general, there are two cases for the underlying model architecture:

1. The model has an easily accessible final layer that we can manipulate
2. The model's ``forward`` method has some non-module function

In the first case, which we demonstrate below, we can take a copy of this last layer as the second part of our wrapper and set it to the ``Identity`` module in the first part of the wrapper.
The second case is a bit more involved, but an example of how we handle this can be seen in our :py:class:`ViSNet <mtenn.conversion_utils.visnet.ViSNet>` implementation.

We'll start with showing the overall skeleton of what the new class will look like, and then go in depth to describe each function.

Overall Structure
^^^^^^^^^^^^^^^^^

We will first want to create a new file in the ``mtenn/conversion_utils/`` directory, which we'll call ``schnet.py``.
Inside this file we'll define the following class:

.. code-block:: python

    from copy import deepcopy
    import torch
    from torch_geometric.nn.models import SchNet as PygSchNet

    from mtenn.model import GroupedModel, Model
    from mtenn.strategy import ComplexOnlyStrategy, ConcatStrategy, DeltaStrategy

    class SchNet(PygSchNet):

        def __init__(self, *args, model=None, **kwargs):
            ...

        def forward(self, data):
            ...

        def _get_representation(self):
            ...

        def _get_energy_func(self):
            ...

        def _get_delta_strategy(self):
            ...

        def _get_complex_only_strategy(self):
            ...

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
            ...

We make this class a subclass of our underlying model so that we can continue to use the PyG SchNet model's ``forward`` method with minimal code changes.

``__init__``
^^^^^^^^^^^^

We want to allow constructing an ``mtenn`` SchNet model either by passing a reference PyG SchNet model, or by passing arguments that can be passed to the PyG SchNet constructor.

.. code-block:: python

    def __init__(self, *args, model=None, **kwargs):
        """
        Initialize the underlying torch_geometric SchNet model.

        Parameters
        ----------
        model : torch_geometric.nn.models.SchNet, optional
            PyTorch Geometric SchNet model to use to construct the underlying model
        """
        # If no model is passed, pass args to torch_geometric, otherwise copy
        #  all parameters and weights over
        if model is None:
            super(SchNet, self).__init__(*args, **kwargs)
        else:
            try:
                # Make sure the atomref for our model is a separate tensor
                atomref = model.atomref.weight.detach().clone()
            except AttributeError:
                atomref = None
            # Extract params from the model
            model_params = (
                model.hidden_channels,
                model.num_filters,
                model.num_interactions,
                model.num_gaussians,
                model.cutoff,
                model.interaction_graph,
                model.interaction_graph.max_num_neighbors,
                model.readout,
                model.dipole,
                model.mean,
                model.std,
                atomref,
            )
            # Construct new model and copy over weights
            super(SchNet, self).__init__(*model_params)
            self.load_state_dict(model.state_dict())

``forward``
^^^^^^^^^^^

As previously mentioned, we want to use the PyG SchNet model's ``forward`` method so we don't have to rewrite any of their code.
The only code that we have to write for our ``forward`` method is to unpack the input data.
All ``mtenn`` models expect to receive data in the form of a ``dict``, which will sometimes need to be unpacked in order to be passed to the underlying models.
In this case, we expect a ``dict`` with keys ``"pos"``, containing the atomic positions as a tensor, and  ``"z"``, containing the atomic numbers as a tensor.

Note that in general, calling an object's ``forward`` method directly is not recommended, however as long as the ``mtenn`` model is called as ``model(...)``, the appropriate ``torch`` hooks should still be called.

.. code-block:: python

    def forward(self, data):
        """
        Make a prediction of the target property based on an input structure.

        Parameters
        ----------
        data : dict[str, torch.Tensor]
            This dictionary should at minimum contain entries for:
            * "pos": Atom coordinates
            * "z": Atomic numbers

        Returns
        -------
        torch.Tensor
            Model prediction
        """
        return super(SchNet, self).forward(data["z"], data["pos"])

``_get_representation``
^^^^^^^^^^^^^^^^^^^^^^^

This is the method responsible for modifying a copy of the calling model such that it can work as a ``Representation`` block, ie it takes a structure as input and returns an n-dimensional vector representation.

The PyG SchNet model has a final single linear layer that goes from a hidden representation to a scalar value.
This hidden representation is exactly what we want as an output, so we can simply set this last linear layer to instead be an ``Identity`` module, which will just pass through the representation.

.. code-block:: python

    def _get_representation(self):
        """
        Copy model and set last layer as an Identity.

        Parameters
        ----------
        model: mtenn.conversion_utils.schnet.SchNet
            SchNet model

        Returns
        -------
        mtenn.conversion_utils.schnet.SchNet
            Copied SchNet model with the last layer replaced by Identity
        """

        # Copy model so initial model isn't affected
        model_copy = deepcopy(self)
        # Replace final linear layer with an identity module
        model_copy.lin2 = torch.nn.Identity()

        return model_copy

``_get_energy_func``
^^^^^^^^^^^^^^^^^^^^

This method is responsible for creating a callable module that can be used inside of a ``Strategy`` block.
This will be used for any ``Strategy`` that requires predicting a scalar value from a single representation, eg :py:class:`DeltaStrategy <mtenn.strategy.DeltaStrategy>`.

For this example, we can simply take a copy of the linear layer that we replaced with an ``Identity`` module in the previous function.
This works because we copied the model in the previous function, so the original final linear layer remains intact.

.. code-block:: python

    def _get_energy_func(self):
        """
        Return copy of last layer of the model.

        Parameters
        ----------
        model: mtenn.conversion_utils.schnet.SchNet
            SchNet model

        Returns
        -------
        torch.nn.modules.linear.Linear
            Copy of last layer
        """

        return deepcopy(self.lin2)

``_get_delta_strategy`` and ``_get_complex_only_strategy``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These next two methods build the respective ``Strategy`` blocks, calling the above ``_get_energy_func`` to construct their required inputs.
See their  docs pages in :py:mod:`mtenn.strategy` for more information on the individual ``Strategy`` blocks.
Any new ``Strategy`` that is implemented may need a corresponding function in each :py:mod:`mtenn.conversion_utils` class.

.. code-block:: python

    def _get_delta_strategy(self):
        """
        Build a mtenn.strategy.DeltaStrategy object based on the calling model.

        Returns
        -------
        mtenn.strategy.DeltaStrategy
            DeltaStrategy built from the model
        """

        return DeltaStrategy(self._get_energy_func())

    def _get_complex_only_strategy(self):
        """
        Build a mtenn.strategy.ComplexOnlyStrategy object based on the calling model.

        Returns
        -------
        mtenn.strategy.ComplexOnlyStrategy
            ComplexOnlyStrategy built from the model
        """

        return ComplexOnlyStrategy(self._get_energy_func())

``get_model``
^^^^^^^^^^^^^

This final method is responsible for taking an :py:mod:`mtenn.conversion_utils` class instance and turning it into an appropriate :py:mod:`mtenn.model` class instance.
This function will need to be updated as well for any new ``Strategy`` types that are added.

.. code-block:: python

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
        Exposed function to build an mtenn.model.Model or mtenn.model.GroupedModel from
        an mtenn.conversion_utils.schnet.SchNet (or args/kwargs). If no model isgiven,
        build a default SchNet model.

        Parameters
        ----------
        model: mtenn.conversion_utils.schnet.SchNet, optional
            SchNet model to use to build the Model object. If not given, build a
            default model
        grouped: bool, default=False
            Build a GroupedModel
        fix_device: bool, default=False
            If True, make sure the input is on the same device as the model,
            copying over as necessary
        strategy: str, default='delta'
            Strategy to use to combine representations of the different parts.
            Options are [delta, concat, complex]
        combination: mtenn.combination.Combination, optional
            Combination object to use to combine multiple predictions. A value must
            be passed if grouped is True
        pred_readout : mtenn.readout.Readout, optional
            Readout object for the individual energy predictions. If a
            GroupedModel is being built, this Readout will be applied to each
            individual prediction before the values are passed to the Combination.
            If a Model is being built, this will be applied to the single prediction
            before it is returned
        comb_readout : mtenn.readout.Readout, optional
            Readout object for the combined multi-pose prediction, in the case that a
            GroupedModel is being built. Otherwise, this is ignored

        Returns
        -------
        mtenn.model.Model
            Model or GroupedModel containing the desired Representation,
            Strategy, and Combination and Readouts as desired
        """
        if model is None:
            model = SchNet()

        # First get representation module
        representation = model._get_representation()

        # Construct strategy module based on model and
        #  representation (if necessary)
        strategy = strategy.lower()
        if strategy == "delta":
            strategy = model._get_delta_strategy()
        elif strategy == "concat":
            strategy = ConcatStrategy()
        elif strategy == "complex":
            strategy = model._get_complex_only_strategy()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Check on combination
        if grouped and (combination is None):
            raise ValueError(
                "Must pass a value for combination if grouped is True."
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

.. _new-inst-model-config:

Adding the new model to :py:mod:`mtenn.config`
----------------------------------------------

After implementing the model in :py:mod:`mtenn.conversion_utils`, you must then add an entry for it in :py:mod:`mtenn.config`.
This is generally a simple process, and mainly just consists of creating a Pydantic schema that defines all the available hyperparameters.

Before beginning with our class, we need to add the model as a possible type in :py:class:`mtenn.config.ModelType`.
This is as simple as ading the line ``schnet = "schnet"`` in our case (or ``my_model = "my_model"`` generally) in the :py:class:`mtenn.config.ModelType` enum.

Our new model config class will subclass the :py:class:`mtenn.config.ModelConfigBase` class.
Because we are using Pydantic, we don't need to define an ``__init__`` function.
Instead, we simply list the hyperparameters and their defaults as Pydantic ``Field``s.
In ``mtenn/config.py``:

.. code-block:: python

    class SchNetModelConfig(ModelConfigBase):
        """
        Class for constructing a SchNet ML model. Default values here are the default values
        given in PyG.
        """

        model_type: ModelType = Field(ModelType.schnet, const=True)

        hidden_channels: int = Field(128, description="Hidden embedding size.")
        num_filters: int = Field(
            128, description="Number of filters to use in the cfconv layers."
        )
        num_interactions: int = Field(6, description="Number of interaction blocks.")
        num_gaussians: int = Field(
            50, description="Number of gaussians to use in the interaction blocks."
        )
        interaction_graph: Callable | None = Field(
            None,
            description=(
                "Function to compute the pairwise interaction graph and "
                "interatomic distances."
            ),
        )
        cutoff: float = Field(
            10, description="Cutoff distance for interatomic interactions."
        )
        max_num_neighbors: int = Field(
            32, description="Maximum number of neighbors to collect for each node."
        )
        readout: str = Field(
            "add", description="Which global aggregation to use [add, mean]."
        )
        dipole: bool = Field(
            False,
            description=(
                "Whether to use the magnitude of the dipole moment to make the "
                "final prediction."
            ),
        )
        mean: float | None = Field(
            None,
            description=(
                "Mean of property to predict, to be added to the model prediction before "
                "returning. This value is only used if dipole is False and a value is also "
                "passed for std."
            ),
        )
        std: float | None = Field(
            None,
            description=(
                "Standard deviation of property to predict, used to scale the model "
                "prediction before returning. This value is only used if dipole is False "
                "and a value is also passed for mean."
            ),
        )
        atomref: list[float] | None = Field(
            None,
            description=(
                "Reference values for single-atom properties. Should have length of 100 to "
                "match with PyG."
            ),
        )

In addition to the hyperparameters, we also have a ``model_type`` constant, which lets us identify which model this config defines in the future.

Inside this class, we also need to define any validators on the hyperparameters.
The base :py:class:`mtenn.config.ModelConfigBase` class implements a ``_check_grouped`` validator that ensure that if we are building a :py:class:`GroupedModel <mtenn.model.GroupedModel>`, all the appropriate options are set.
In our validator, we'll also make sure the provided ``atomref`` is the right size.
Note that the below code is indented as if it were in the top level of the file, but it should be a method of the above class.

.. code-block:: python

    @root_validator(pre=False)
    def validate(cls, values):
        """
        values is a dict of the parsed pydantic Fields, that gets passed in
        automatically
        """
        # Make sure the grouped stuff is properly assigned
        ModelConfigBase._check_grouped(values)

        # Make sure atomref length is correct (this is required by PyG)
        atomref = values["atomref"]
        if (atomref is not None) and (len(atomref) != 100):
            raise ValueError(f"atomref must be length 100 (got {len(atomref)})")

        return values

Other than the validators, the only thing that we need to implement is the ``_build`` function, which will get called automatically by the ``ModelConfigBase.build`` function.
In this function we will first build an :py:class:`mtenn.conversion_utils.schnet.SchNet` model, and then use that model, along with the ``mtenn_params`` passed by the ``ModelConfigBase.build`` function, to build an :py:class:`mtenn.model` model.
As above, this function should be a method of the ``SchNetModelConfig`` class.

.. code-block:: python

    def _build(self, mtenn_params={}):
        """
        Build an mtenn SchNet Model from this config.

        :meta public:

        Parameters
        ----------
        mtenn_params : dict, optional
            Dictionary that stores the Readout objects for the individual
            predictions and for the combined prediction, and the Combination object
            in the case of a multi-pose model. These are all constructed the same for all
            Model types, so we can just handle them in the base class. Keys in the
            dict will be:

            * "combination": mtenn.combination.Combination

            * "pred_readout": mtenn.readout.Readout for individual pose predictions

            * "comb_readout": mtenn.readout.Readout for combined prediction (in the case
            of a multi-pose model)

        Returns
        -------
        mtenn.model.Model
            Model constructed from the config
        """
        from mtenn.conversion_utils.schnet import SchNet

        # Create an mtenn SchNet model from PyG SchNet model
        model = SchNet(
            hidden_channels=self.hidden_channels,
            num_filters=self.num_filters,
            num_interactions=self.num_interactions,
            num_gaussians=self.num_gaussians,
            interaction_graph=self.interaction_graph,
            cutoff=self.cutoff,
            max_num_neighbors=self.max_num_neighbors,
            readout=self.readout,
            dipole=self.dipole,
            mean=self.mean,
            std=self.std,
            atomref=self.atomref,
        )

        combination = mtenn_params.get("combination", None)
        pred_readout = mtenn_params.get("pred_readout", None)
        comb_readout = mtenn_params.get("comb_readout", None)

        return SchNet.get_model(
            model=model,
            grouped=self.grouped,
            fix_device=True,
            strategy=self.strategy,
            combination=combination,
            pred_readout=pred_readout,
            comb_readout=comb_readout,
        )

We can now build a default SchNet :py:class:`mtenn.model.Model` with:

.. code-block:: python

    from mtenn.config import SchNetModelConfig
    config = SchNetModelConfig()
    model = config.build()
