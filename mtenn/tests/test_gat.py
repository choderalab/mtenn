import pytest
import torch

from dgllife.model import GAT as GAT_dgl
from dgllife.utils import CanonicalAtomFeaturizer, SMILESToBigraph
from mtenn.conversion_utils.gat import GAT


@pytest.fixture
def model_input():
    smiles = "CCCC"
    g = SMILESToBigraph(add_self_loop=True, node_featurizer=CanonicalAtomFeaturizer())(
        smiles
    )

    return {"g": g, "smiles": smiles}


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


def test_gat_can_predict(model_input):
    model = GAT(in_feats=CanonicalAtomFeaturizer().feat_size())
    _ = model(model_input)


def test_representation_is_correct():
    model = GAT(in_feats=10)
    rep = model._get_representation()

    model_params = dict(model.named_parameters())
    for n, rep_param in rep.named_parameters():
        assert torch.allclose(rep_param, model_params[n])


def test_get_model_no_ref():
    model = GAT.get_model(in_feats=10)

    assert isinstance(model.representation, GAT)
    assert model.readout is None


def test_get_model_ref():
    ref_model = GAT(in_feats=10)
    model = GAT.get_model(model=ref_model)

    assert isinstance(model.representation, GAT)
    assert model.readout is None

    ref_params = dict(ref_model.named_parameters())
    for n, model_param in model.representation.named_parameters():
        assert torch.allclose(model_param, ref_params[n])
