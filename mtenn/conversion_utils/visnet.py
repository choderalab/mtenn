"""
Representation and strategy for ViSNet model.
"""
import warnings
from copy import deepcopy
import torch
from torch.autograd import grad
from torch_geometric.utils import scatter

from mtenn.model import GroupedModel, Model
from mtenn.strategy import ComplexOnlyStrategy, ConcatStrategy, DeltaStrategy

# guard required: currently require PyG nightly 2.5.0 (SEE ISSUE #42)
HAS_VISNET = False

try:
    from torch_geometric.nn.models import ViSNet as PygVisNet
    from torch_geometric.nn.models.visnet import ViS_MP_Vertex
    HAS_VISNET = True
except ImportError:
    warnings.warn("VisNet import error. Is your PyG >=2.5.0? Refer to issue #42", ImportWarning)

class EquivariantVecToScalar(torch.nn.Module):
    # Wrapper for PygVisNet.EquivariantScalar to implement forward() method
    def __init__(self, mean, reduce_op):
        super(EquivariantVecToScalar, self).__init__()
        self.mean = mean
        self.reduce_op = reduce_op
    def forward(self, x):
        # dummy variable. all atoms from the same molecule and the same batch
        batch = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device)
        y = scatter(x, batch, dim=0, reduce=self.reduce_op)
        return y + self.mean


if HAS_VISNET:
    class ViSNet(torch.nn.Module):
        def __init__(self, *args, model=None, **kwargs):
            super().__init__()
            ## If no model is passed, construct default ViSNet model, otherwise copy
            ##  all parameters and weights over
            if model is None:
                self.visnet = PygVisNet(*args, **kwargs)
            else:
                atomref = model.prior_model.atomref.weight.detach().clone()
                model_params = {
                    'lmax': model.representation_model.lmax,
                    'vecnorm_type': model.representation_model.vecnorm_type,
                    'trainable_vecnorm': model.representation_model.trainable_vecnorm,
                    'num_heads': model.representation_model.num_heads,
                    'num_layers': model.representation_model.num_layers,
                    'hidden_channels': model.representation_model.hidden_channels,
                    'num_rbf': model.representation_model.num_rbf,
                    'trainable_rbf': model.representation_model.trainable_rbf,
                    'max_z': model.representation_model.max_z,
                    'cutoff': model.representation_model.cutoff,
                    'reduce_op': model.representation_model.max_num_neighbors,
                    'vertex': isinstance(model.representation_model.vis_mp_layers[0], ViS_MP_Vertex),
                    'reduce_op': model.reduce_op,
                    'mean': model.mean,
                    'std': model.std,
                    'derivative': model.derivative, # not used. originally calculates "force" from energy
                    'atomref': atomref,
                }
                self.visnet = PygVisNet(**model_params)
                self.visnet.load_state_dict(model.state_dict())

            self.readout = EquivariantVecToScalar(self.visnet.mean, self.visnet.reduce_op)

        def forward(self, data):
            """
            Computes the vector representation of a molecule from its atomic numbers and positional coordinates, to produce a vector representation. The output vector can be used for further molecular analysis or prediction tasks.

            Parameters
            ----------
            data : dict[str, torch.Tensor]
                A dictionary containing the atomic point clouds of a molecule. It should have the following keys:
                - z (torch.Tensor): The atomic numbers of the atoms in the molecule.
                - pos (torch.Tensor): The 3D coordinates (x, y, z) of each atom in the molecule.

            Returns
            -------
            torch.Tensor
                A tensor representing the vector output of the molecule

            Notes
            -----
            Assumes all atoms passed in `data` belong to the same molecule and are processed in a single batch. 
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
            Input model, remove last layer.

            Returns
            -------
            ViSNet
                Copied ViSNet model, removing the last MLP layer that takes vector representation to scalar output.
            """

            ## Copy model so initial model isn't affected
            model_copy = deepcopy(self)

            return model_copy

        def _get_energy_func(self):
            """
            Return last layer of the model (outputs scalar value)

            Returns
            -------
            torch.nn.Module
                Copy of `model`'s last layer, which is an instance of EquivariantVecToScalar() class
            """
            return deepcopy(self.readout)

        def _get_delta_strategy(self):
            """
            Build a DeltaStrategy object based on the passed model.

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
            Exposed function to build a Model object from a VisNet object. If none
            is provided, a default model is initialized.

            Parameters
            ----------
            model: VisNet, optional
                VisNet model to use to build the Model object. If left as none, a default model will be initialized and used
            grouped: bool, default=False
                Whether this model should accept groups of inputs or one input at a time.
            fix_device: bool, default=False
                If True, make sure the input is on the same device as the model,
                copying over as necessary.
            strategy: str, default='delta'
                Strategy to use to combine representation of the different parts.
                Options are ['delta', 'concat', 'complex']
            combination: Combination, optional
                Combination object to use to combine predictions in a group. A value must be passed if `grouped` is `True`.
            pred_readout : Readout
                Readout object for the energy predictions. If `grouped` is `False`, this option will still be used in the construction of the `Model` object.
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
