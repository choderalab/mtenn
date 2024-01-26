"""
Representation and strategy for ViSNet model.
"""
from copy import deepcopy
import torch
from torch.autograd import grad
from torch_geometric.utils import scatter

from mtenn.model import GroupedModel, Model
from mtenn.strategy import ComplexOnlyStrategy, ConcatStrategy, DeltaStrategy

HAS_VISNET_FLAG = False
try:
    from torch_geometric.nn.models import ViSNet as PygVisNet
    HAS_VISNET_FLAG = True
except ImportError:
    pass    

class EquivariantVecToScaler(torch.nn.Module):
    # Wrapper for PygVisNet.EquivariantScalar to implement forward() method
    def __init__(self, mean, reduce_op):
        super(EquivariantVecToScaler, self).__init__()
        self.mean = mean
        self.reduce_op = reduce_op
    def forward(self, x):
        # dummy variable. all atoms from the same molecule and the same batch
        batch = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device)

        y = scatter(x, batch, dim=0, reduce=self.reduce_op)
        return y + self.mean


class ViSNet(torch.nn.Module):
    def __init__(self, *args, model=None, **kwargs):
        super().__init__()
        ## If no model is passed, construct default SchNet model, otherwise copy
        ##  all parameters and weights over
        if model is None:
            self.visnet = PygVisNet(*args, **kwargs)
        else:
            try:
                atomref = model.atomref.weight.detach().clone()
            except AttributeError:
                atomref = None
            model_params = (
                model.lmax,
                model.vecnorm_type,
                model.trainable_vecnorm,
                model.num_heads,
                model.num_layers,
                model.hidden_channels,
                model.num_rbf,
                model.trainable_rbf,
                model.max_z,
                model.cutoff,
                model.max_num_neighbors,
                model.vertex,
                model.reduce_op,
                model.mean,
                model.std,
                model.derivative, # not used. originally calculates "force" from energy
                atomref,
            )
            self.visnet = PygVisNet(*model_params)
            self.load_state_dict(model.state_dict())

        self.readout = EquivariantVecToScaler(self.visnet.mean, self.visnet.reduce_op)

    def forward(self, data):
        """
        Computes the energies or properties (forces) for a batch of
        molecules.

        Args:
            z (torch.Tensor): The atomic numbers.
            pos (torch.Tensor): The coordinates of the atoms.
            batch (torch.Tensor): A batch vector,
                which assigns each node to a specific example.

        Returns:
            x (torch.Tensor): Scalar output based on node features and vector features.
        """
        pos = data["pos"]
        z = data["z"]

        # all atom in one pass from the same molecule
        # TODO: set separate batch for ligand and protein
        batch = torch.zeros(z.shape[0], device=z.device)
        x, v = self.visnet.representation_model(z, pos, batch)
        x = self.visnet.output_model.pre_reduce(x, v)
        x = x * self.visnet.std

        return x

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

        return model_copy

    def _get_energy_func(self):
        """
        Return last layer of the model (outputs scalar value)

        Parameters
        ----------
        model: SchNet
            SchNet model

        Returns
        -------
        torch.nn.modules.linear.Linear
            Copy of `model`'s last layer
        """
        return deepcopy(self.readout)

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

    def _get_complex_only_strategy(self):
        """
        Build a ComplexOnlyStrategy object based on the passed model.

        Returns
        -------
        ComplexOnlyStrategy
            ComplexOnlyStrategy built from `self`
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
        if model is None:
            model = ViSNet()

        ## First get representation module
        representation = model._get_representation()

        ## Construct strategy module based on model and
        ##  representation (if necessary)
        strategy = strategy.lower()
        if strategy == "delta":
            strategy = model._get_delta_strategy()
        elif strategy == "concat":
            strategy = ConcatStrategy()
        elif strategy == "complex":
            strategy = model._get_complex_only_strategy()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

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
