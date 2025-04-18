"""
``Representation`` and ``Strategy`` implementations for the graph attention model
architecture. The underlying model that we use is the implementation in the
`DGL-LifeSCi <https://lifesci.dgl.ai/api/model.gnn.html#module-dgllife.model.gnn.gat>`_
package.
"""
from copy import deepcopy
import torch
from dgllife.model import GAT as GAT_dgl
from dgllife.model import WeightedSumAndMax

from mtenn.model import LigandOnlyModel


class GAT(torch.nn.Module):
    """
    ``mtenn`` wrapper around the DGL-LifeSci GAT model. This class handles construction
    of the model and the formatting into ``Representation`` and ``Strategy`` blocks.
    """

    def __init__(self, *args, model=None, **kwargs):
        """
        Initialize the underlying ``dgllife.model.GAT`` model, as well as the ``mtenn``
        -specific code on top. If a value is passed for ``model``, builds a new
        ``dgllife.model.GAT`` model based on those hyperparameters, and copies over the
        weights. Otherwise, all ``*args`` and ``**kwargs`` are passed directly to the
        ``dgllife.model.GAT`` constructor.

        Parameters
        ----------
        model : ``dgllife.model.GAT``, optional
            DGL-LifeSci model to use to construct the underlying model
        """
        super().__init__()

        # First check for predictor_hidden_feats so it doesn't get passed to DGL GAT
        #  constructor
        predictor_hidden_feats = kwargs.pop("predictor_hidden_feats", None)

        # If no model is passed, construct model based on passed args, otherwise copy
        #  all parameters and weights over
        if model is None:
            self.gnn = GAT_dgl(*args, **kwargs)
        else:
            # Parameters that are conveniently accessible from the top level
            in_feats = model.gnn_layers[0].gat_conv.fc.in_features
            hidden_feats = model.hidden_feats
            num_heads = model.num_heads
            agg_modes = model.agg_modes
            # Parameters that can only be adcessed layer-wise
            layer_params = []
            for l in model.gnn_layers:
                gc = l.gat_conv
                new_params = (
                    gc.feat_drop.p,
                    gc.attn_drop.p,
                    gc.leaky_relu.negative_slope,
                    gc.activation,
                    bool(gc.res_fc),
                    (gc.res_fc.bias is not None)
                    if gc.has_linear_res
                    else gc.has_explicit_bias,
                )
                layer_params += [new_params]

            (
                feat_drops,
                attn_drops,
                alphas,
                activations,
                residuals,
                biases,
            ) = zip(*layer_params)
            self.gnn = GAT_dgl(
                in_feats=in_feats,
                hidden_feats=hidden_feats,
                num_heads=num_heads,
                feat_drops=feat_drops,
                attn_drops=attn_drops,
                alphas=alphas,
                residuals=residuals,
                agg_modes=agg_modes,
                activations=activations,
                biases=biases,
            )
            self.gnn.load_state_dict(model.state_dict())

        # Copied from GATPredictor class, figure out how many features the last
        #  layer of the GNN will have
        if self.gnn.agg_modes[-1] == "flatten":
            gnn_out_feats = self.gnn.hidden_feats[-1] * self.gnn.num_heads[-1]
        else:
            gnn_out_feats = self.gnn.hidden_feats[-1]
        # Returns a Tensor 2x length of input (sum and max)
        self.readout = WeightedSumAndMax(gnn_out_feats)
        self.readout_out_feats = 2 * gnn_out_feats

        # Use given hidden feats if supplied, otherwise use 1/2 gnn_out_feats
        if predictor_hidden_feats is None:
            predictor_hidden_feats = gnn_out_feats // 2

        # 2 layer MLP with ReLU activation (borrowed from GATPredictor)
        self.predict = torch.nn.Sequential(
            torch.nn.Linear(self.readout_out_feats, predictor_hidden_feats),
            torch.nn.ReLU(),
            torch.nn.Linear(predictor_hidden_feats, 1),
        )

    def forward(self, data):
        """
        Make a prediction of the target property based on an input molecule graph.

        Parameters
        ----------
        data : dict
            This dictionary should at minimum contain an entry for ``"g"``, which should
            be the molecule graph representation and will be passed to the underlying
            ``dgllife.model.GAT`` object

        Returns
        -------
        torch.Tensor
            Model prediction
        """
        g = data["g"]
        node_feats = self.gnn(g, g.ndata["h"])
        graph_feats = self.readout(g, node_feats)
        return self.predict(graph_feats)

    @property
    def output_dim(self):
        return self.readout_out_feats

    @property
    def extract_key(self):
        return None

    def _get_representation(self):
        """
        Input model, remove last layer.

        Returns
        -------
        GAT
            Copied GAT model with the last layer replaced by an Identity module
        """

        # Copy model so initial model isn't affected
        model_copy = deepcopy(self)

        # Replace predict module with identity module
        model_copy.predict = torch.nn.Identity()

        return model_copy

    def _get_energy_func(self, layer_norm=False):
        """
        Return last two layer of the model.

        Returns
        -------
        torch.nn.Sequential
            Sequential module calling copy of `model`'s last two layers
        layer_norm: bool, default=False
            Apply a ``LayerNorm`` normalization before passing through the model's
            predictor
        """

        pred_model = deepcopy(self.predict)
        if layer_norm:
            energy_func = torch.nn.Sequential(
                torch.nn.LayerNorm(self.output_dim), pred_model
            )
        else:
            energy_func = pred_model

        return energy_func

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
