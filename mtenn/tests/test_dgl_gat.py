try:
    from dgllife.model import GAT as GAT_dgl
    from dgllife.utils import CanonicalAtomFeaturizer, SMILESToBigraph
    from mtenn.conversion_utils.dgl_gat import DGLGAT
except ModuleNotFoundError:
    import pytest

    pytest.skip("DGL not found, skipping DGL GAT tests.", allow_module_level=True)

import pytest
import torch


@pytest.fixture
def model_input():
    smiles = "CCCC"
    g = SMILESToBigraph(add_self_loop=True, node_featurizer=CanonicalAtomFeaturizer())(
        smiles
    )

    return {"g": g, "smiles": smiles}


def test_build_gat_directly_kwargs():
    model = DGLGAT(in_feats=10, hidden_feats=[1, 2, 3])
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
    model = DGLGAT(model=dgl_model)

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
    model = DGLGAT(in_feats=10, predictor_hidden_feats=10)
    assert model.predict[0].out_features == 10


def test_gat_can_predict(model_input):
    model = DGLGAT(in_feats=CanonicalAtomFeaturizer().feat_size())
    _ = model(model_input)


def test_representation_is_correct():
    model = DGLGAT(in_feats=10)
    rep = model._get_representation()

    model_params = dict(model.named_parameters())
    for n, rep_param in rep.named_parameters():
        assert torch.allclose(rep_param, model_params[n])


def test_get_model_no_ref():
    model = DGLGAT.get_model(in_feats=10)

    assert isinstance(model.representation, DGLGAT)
    assert model.readout is None


def test_get_model_ref():
    ref_model = DGLGAT(in_feats=10)
    model = DGLGAT.get_model(model=ref_model)

    assert isinstance(model.representation, DGLGAT)
    assert model.readout is None

    ref_params = dict(ref_model.named_parameters())
    for n, model_param in model.representation.named_parameters():
        assert torch.allclose(model_param, ref_params[n])
