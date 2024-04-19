"""
``Representation`` and ``Strategy`` implementations for the ViSNet model architecture.
The underlying model that we use is the implementation in
`PyTorch Geometric <https://pytorch-geometric.readthedocs.io/en/latest/generated/
torch_geometric.nn.models.ViSNet.html#torch_geometric.nn.models.ViSNet>`_.
"""
from copy import deepcopy
import torch
from torch_geometric.utils import scatter
from torch_geometric.nn.models import ViSNet as PygViSNet
from torch_geometric.nn.models.visnet import ViS_MP_Vertex

from mtenn.model import GroupedModel, Model
from mtenn.strategy import ComplexOnlyStrategy, ConcatStrategy, DeltaStrategy


class EquivariantVecToScalar(torch.nn.Module):
    """
    Wrapper around ``torch_geometric.utils.scatter`` to use it as a ``Module``.
    """

    def __init__(self, mean, reduce_op):
        """
        Store use parameters.

        Parameters
        ----------
        mean : torch.Tensor
            Mean of predicted value
        reduce_op : str
            Reduce operation to use in ``torch_geometric.utils.scatter``
        """
        super(EquivariantVecToScalar, self).__init__()
        self.mean = mean
        self.reduce_op = reduce_op

    def forward(self, x):
        """
        Perform the scatter operation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Output of ``scatter`` call
        """
        # All atoms from the same molecule and the same batch
        batch = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device)
        y = scatter(x, batch, dim=0, reduce=self.reduce_op)
        return y + self.mean


class ViSNet(torch.nn.Module):
    """
    ``mtenn`` wrapper around the PyTorch Geometric ViSNet model. This class handles
    construction of the model and the formatting into ``Representation`` and
    ``Strategy`` blocks.
    """

    def __init__(self, *args, model=None, **kwargs):
        """
        Initialize the underlying ``torch_geometric.nn.models.ViSNet`` model. If a value
        is passed for ``model``, builds a new ``torch_geometric.nn.models.ViSNet`` model
        based on those hyperparameters, and copies over the weights. Otherwise, all
        ``*args`` and ``**kwargs`` are passed directly to the
        ``torch_geometric.nn.models.ViSNet`` constructor.

        Parameters
        ----------
        model : ``torch_geometric.nn.models.ViSNet``, optional
            PyTorch Geometric ViSNet model to use to construct the underlying model
        """
        super().__init__()
        # If no model is passed, construct default ViSNet model, otherwise copy
        #  all parameters and weights over
        if model is None:
            self.visnet = PygViSNet(*args, **kwargs)
        else:
            atomref = model.prior_model.atomref.weight.detach().clone()
            model_params = {
                "lmax": model.representation_model.lmax,
                "vecnorm_type": model.representation_model.vecnorm_type,
                "trainable_vecnorm": model.representation_model.trainable_vecnorm,
                "num_heads": model.representation_model.num_heads,
                "num_layers": model.representation_model.num_layers,
                "hidden_channels": model.representation_model.hidden_channels,
                "num_rbf": model.representation_model.num_rbf,
                "trainable_rbf": model.representation_model.trainable_rbf,
                "max_z": model.representation_model.max_z,
                "cutoff": model.representation_model.cutoff,
                "max_num_neighbors": model.representation_model.max_num_neighbors,
                "vertex": isinstance(
                    model.representation_model.vis_mp_layers[0], ViS_MP_Vertex
                ),
                "reduce_op": model.reduce_op,
                "mean": model.mean,
                "std": model.std,
                "derivative": model.derivative,  # not used. originally calculates "force" from energy
                "atomref": atomref,
            }
            self.visnet = PygViSNet(**model_params)
            self.visnet.load_state_dict(model.state_dict())

        self.readout = EquivariantVecToScalar(self.visnet.mean, self.visnet.reduce_op)

    def forward(self, data):
        """
        Predict a vector representation of an input structure.

        Parameters
        ----------
        data : dict[str, torch.Tensor]
            This dictionary should at minimum contain entries for:

            * ``"pos"``: Atom coordinates

            * ``"z"``: Atomic numbers

        Returns
        -------
        torch.Tensor
            Predicted vector representation of input
        """
        pos = data["pos"]
        z = data["z"]

        # all atom in one pass from the same molecule
        batch = torch.zeros(z.shape[0], device=z.device)
        x, v = self.visnet.representation_model(z, pos, batch)
        x = self.visnet.output_model.pre_reduce(x, v)
        x = x * self.visnet.std
        if self.visnet.prior_model is not None:
            x = self.visnet.prior_model(x, z)

        return x

    def _get_representation(self):
        """
        Copy model.

        Returns
        -------
        mtenn.conversion_utils.visnet.ViSNet
            Copied ``ViSNet`` model
        """
        # Copy model so initial model isn't affected
        return deepcopy(self)

    def _get_energy_func(self):
        """
        Return copy of ``readout`` portion of the model.

        Returns
        -------
        mtenn.conversion_utils.visnet.EquivariantVecToScalar
            Copy of ``self.readout``
        """
        return deepcopy(self.readout)

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
        Exposed function to build a :py:class:`Model <mtenn.model.Model>` or
        :py:class:`GroupedModel <mtenn.model.GroupedModel>` from a :py:class:`ViSNet
        <mtenn.conversion_utils.visnet.ViSNet>` (or args/kwargs). If no ``model`` is
        given, build a default ``ViSNet`` model.

        Parameters
        ----------
        model: mtenn.conversion_utils.visnet.ViSNet, optional
            ``ViSNet`` model to use to build the ``Model`` object. If not given, build a
            default model
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
            model = ViSNet()

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
