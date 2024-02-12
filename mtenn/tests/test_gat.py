from dgllife.model import GAT as GAT_dgl
from dgllife.utils import CanonicalAtomFeaturizer, SMILESToBigraph
from mtenn.conversion_utils.gat import GAT


def test_build_gat_directly_kwargs():
    model = GAT(in_feats=10, hidden_feats=[1, 2, 3])
    assert len(model.gnn.gnn_layers) == 3

    assert model.gnn.gnn_layers[0].gat_conv._in_src_feats == 10
    assert model.gnn.gnn_layers[0].gat_conv._out_feats == 1

    # hidden_feats * num_heads = 1 * 4
    assert model.gnn.gnn_layers[1].gat_conv._in_src_feats == 4
    assert model.gnn.gnn_layers[1].gat_conv._out_feats == 2

    # hidden_feats * num_heads = 2 * 4
    assert model.gnn.gnn_layers[2].gat_conv._in_src_feats == 8
    assert model.gnn.gnn_layers[2].gat_conv._out_feats == 3


def test_build_gat_from_dgl_gat():
    dgl_model = GAT_dgl(in_feats=10, hidden_feats=[1, 2, 3])
    model = GAT(model=dgl_model)

    # Check set up as before
    assert len(model.gnn.gnn_layers) == 3

    assert model.gnn.gnn_layers[0].gat_conv._in_src_feats == 10
    assert model.gnn.gnn_layers[0].gat_conv._out_feats == 1

    # hidden_feats * num_heads = 1 * 4
    assert model.gnn.gnn_layers[1].gat_conv._in_src_feats == 4
    assert model.gnn.gnn_layers[1].gat_conv._out_feats == 2

    # hidden_feats * num_heads = 2 * 4
    assert model.gnn.gnn_layers[2].gat_conv._in_src_feats == 8
    assert model.gnn.gnn_layers[2].gat_conv._out_feats == 3

    # Check that model weights got copied
    ref_params = dict(dgl_model.state_dict())
    for n, model_param in model.gnn.named_parameters():
        assert (model_param == ref_params[n]).all()


def test_set_predictor_hidden_feats():
    model = GAT(in_feats=10, predictor_hidden_feats=10)
    assert model.predict[0].out_features == 10
