"""
``Representation`` and ``Strategy`` implementations for the graph attention model
architecture. The underlying model that we use is the implementation in
`PyTorch Geometric <https://pytorch-geometric.readthedocs.io/en/latest/generated/
torch_geometric.nn.models.GAT.html>`_.
"""
from copy import deepcopy
import torch
from torch_geometric.nn.models import GAT as PygGAT

from mtenn.model import LigandOnlyModel


class GAT(torch.nn.Module):
    """
    ``mtenn`` wrapper around the PyTorch Geometric GAT model. This class handles
    construction of the model and the formatting into ``Representation`` and
    ``Strategy`` blocks.
    """

    def __init__(self, *args, model=None, **kwargs):
        """
        Initialize the underlying ``torch_geometric.nn.models.GAT`` model. If a value is
        passed for ``model``, builds a new ``torch_geometric.nn.models.GAT`` model based
        on those hyperparameters, and copies over the weights. Otherwise, all ``*args``
        and ``**kwargs`` are passed directly to the ``torch_geometric.nn.models.GAT``
        constructor.

        Parameters
        ----------
        model : ``torch_geometric.nn.models.GAT``, optional
            PyTorch Geometric model to use to construct the underlying model
        """
        super().__init__()

        # If no model is passed, construct model based on passed args, otherwise copy
        #  all parameters and weights over
        if model is None:
            self.gnn = PygGAT(*args, **kwargs)
        else:
            self.gnn = deepcopy(model)

        # Predict from mean of node features
        self.predict = torch.nn.Linear(self.gnn.out_channels, 1)

    def forward(self, data):
        """
        Make a prediction of the target property based on an input molecule graph.

        Parameters
        ----------
        data : dict[str, torch.Tensor]
            This dictionary should at minimum contain entries for:

            * ``"x"``: Atom coordinates, shape of (num_atoms, num_features)

            * ``"edge_index"``: All edges in the graph, shape of (2, num_edges) with the
            first row giving the source node indices and the second row giving the
            destination node indices for each edge

        Returns
        -------
        torch.Tensor
            Model prediction
        """
        # Run through GNN
        graph_gred = self.gnn(x=data["x"], edge_index=data["edge_index"])
        # Take mean of feature values across nodes
        graph_gred = graph_gred.mean(dim=0)
        # Make final prediction
        return self.predict(graph_gred)

    def _get_representation(self):
        """
        Input model, remove last layer.

        Returns
        -------
        GAT
            Copied GAT model with the last layer replaced by an Identity module
        """

        # Copy model so initial model isn't affected
        model_copy = deepcopy(self.gnn)

        return model_copy

    def _get_energy_func(self):
        """
        Return last layer of the model.

        Returns
        -------
        torch.nn.Linear
            Final energy prediction layer of the model
        """

        return deepcopy(self.readout)

    @staticmethod
    def get_model(
        *args,
        model=None,
        fix_device=False,
        pred_readout=None,
        **kwargs,
    ):
        """
        Exposed function to build a :py:class:`LigandOnlyModel
        <mtenn.model.LigandOnlyModel>` from a :py:class:`GAT
        <mtenn.conversion_utils.gat.GAT>` (or args/kwargs). If no ``model`` is given,
        use the ``*args`` and ``**kwargs``.

        Parameters
        ----------
        model: mtenn.conversion_utils.gat.GAT, optional
            ``GAT`` model to use to build the ``LigandOnlyModel`` object. If not
            provided, a model will be built using the passed ``*args`` and ``**kwargs``
        fix_device: bool, default=False
            If True, make sure the input is on the same device as the model,
            copying over as necessary
        pred_readout : mtenn.readout.Readout, optional
            ``Readout`` object for the energy predictions

        Returns
        -------
        mtenn.model.LigandOnlyModel
            ``LigandOnlyModel`` object containing the model and desired ``Readout``
        """
        if model is None:
            model = GAT(*args, **kwargs)

        return LigandOnlyModel(model=model, readout=pred_readout, fix_device=fix_device)
