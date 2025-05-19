import pydantic
import pytest

from mtenn.config import (
    GATRepresentationConfig,
    E3NNRepresentationConfig,
    SchNetRepresentationConfig,
)


def test_no_diff_list_lengths_gat():
    with pytest.raises(ValueError):
        # Different length lists should raise error
        _ = GATRepresentationConfig(hidden_feats=[1, 2, 3], num_heads=[4, 5])


def test_bad_param_mapping_gat():
    with pytest.raises(ValueError):
        # Can't convert string to int
        _ = GATRepresentationConfig(hidden_feats="sdf")


def test_can_pass_lists_gat():
    model_config = GATRepresentationConfig(hidden_feats=[1, 2, 3])
    model = model_config.build()

    assert len(model.gnn.gnn_layers) == 3
    assert not model_config._from_num_layers


def test_str_irreps_dict_e3nn():
    irreps_str = "0:10,1:5"
    model_config = E3NNRepresentationConfig(irreps_hidden=irreps_str)
    assert model_config.irreps_hidden == "10x0o+10x0e+5x1o+5x1e"


def test_bad_str_irreps_dict_e3nn():
    irreps_str = "0,1:5"
    with pytest.raises(ValueError):
        _ = E3NNRepresentationConfig(irreps_hidden=irreps_str)


def test_str_irreps_bad_dict_e3nn():
    irreps_str = "0k:10,1:5"
    with pytest.raises(ValueError):
        _ = E3NNRepresentationConfig(irreps_hidden=irreps_str)


def test_str_irreps_str_e3nn():
    irreps_str = "10x0o+10x0e+5x1o+5x1e"
    model_config = E3NNRepresentationConfig(irreps_hidden=irreps_str)
    assert model_config.irreps_hidden == "10x0o+10x0e+5x1o+5x1e"


def test_bad_str_irreps_str_e3nn():
    irreps_str = "10x0k+10x0e+5x1o+5x1e"
    with pytest.raises(ValueError):
        _ = E3NNRepresentationConfig(irreps_hidden=irreps_str)
