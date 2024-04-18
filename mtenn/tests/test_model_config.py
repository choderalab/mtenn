import pydantic
import pytest

from mtenn.config import (
    GATModelConfig,
    E3NNModelConfig,
    SchNetModelConfig,
    ViSNetModelConfig,
)
from mtenn.readout import PIC50Readout, PKiReadout
from mtenn.strategy import ComplexOnlyStrategy, ConcatStrategy, DeltaStrategy


def test_random_seed_gat():
    rand_config = GATModelConfig()
    set_config = GATModelConfig(rand_seed=10)

    rand_model1 = rand_config.build()
    rand_model2 = rand_config.build()
    set_model1 = set_config.build()
    set_model2 = set_config.build()

    rand_equal = [
        (p1 == p2).all()
        for p1, p2 in zip(rand_model1.parameters(), rand_model2.parameters())
    ]
    assert sum(rand_equal) < len(rand_equal)

    set_equal = [
        (p1 == p2).all()
        for p1, p2 in zip(set_model1.parameters(), set_model2.parameters())
    ]
    assert sum(set_equal) == len(set_equal)


@pytest.mark.parametrize(
    "pred_r,pred_r_class,pred_r_args",
    [
        (None, None, [None, None]),
        ("pic50", PIC50Readout, [None, None]),
        ("pic50", PIC50Readout, [0.375, 9.5]),
        ("pki", PKiReadout, [None, None]),
    ],
)
def test_readout_gat(pred_r, pred_r_class, pred_r_args):
    model = GATModelConfig(
        pred_readout=pred_r,
        pred_substrate=pred_r_args[0],
        pred_km=pred_r_args[1],
    ).build()

    if pred_r is None:
        assert model.readout is None
        return

    assert isinstance(model.readout, pred_r_class)
    if pred_r == "pic50":
        assert model.readout.substrate == pred_r_args[0]
        assert model.readout.Km == pred_r_args[1]


def test_model_weights_gat():
    model1 = GATModelConfig().build()
    model2 = GATModelConfig(model_weights=model1.state_dict()).build()

    test_model_params = dict(model2.named_parameters())
    for n, ref_param in model1.named_parameters():
        assert (ref_param == test_model_params[n]).all()


def test_no_diff_list_lengths_gat():
    with pytest.raises(ValueError):
        # Different length lists should raise error
        _ = GATModelConfig(hidden_feats=[1, 2, 3], num_heads=[4, 5])


def test_bad_param_mapping_gat():
    with pytest.raises(ValueError):
        # Can't convert string to int
        _ = GATModelConfig(hidden_feats="sdf")


def test_can_pass_lists_gat():
    model_config = GATModelConfig(hidden_feats=[1, 2, 3])
    model = model_config.build()

    assert len(model.representation.gnn.gnn_layers) == 3
    assert not model_config._from_num_layers


def test_random_seed_e3nn():
    rand_config = E3NNModelConfig()
    set_config = E3NNModelConfig(rand_seed=10)

    rand_model1 = rand_config.build()
    rand_model2 = rand_config.build()
    set_model1 = set_config.build()
    set_model2 = set_config.build()

    rand_equal = [
        (p1 == p2).all()
        for p1, p2 in zip(rand_model1.parameters(), rand_model2.parameters())
    ]
    assert sum(rand_equal) < len(rand_equal)

    set_equal = [
        (p1 == p2).all()
        for p1, p2 in zip(set_model1.parameters(), set_model2.parameters())
    ]
    assert sum(set_equal) == len(set_equal)


@pytest.mark.parametrize(
    "pred_r,pred_r_class,pred_r_args",
    [
        (None, None, [None, None]),
        ("pic50", PIC50Readout, [None, None]),
        ("pic50", PIC50Readout, [0.375, 9.5]),
        ("pki", PKiReadout, [None, None]),
    ],
)
def test_readout_e3nn(pred_r, pred_r_class, pred_r_args):
    model = E3NNModelConfig(
        pred_readout=pred_r,
        pred_substrate=pred_r_args[0],
        pred_km=pred_r_args[1],
    ).build()

    if pred_r is None:
        assert model.readout is None
        return

    assert isinstance(model.readout, pred_r_class)
    if pred_r == "pic50":
        assert model.readout.substrate == pred_r_args[0]
        assert model.readout.Km == pred_r_args[1]


@pytest.mark.parametrize(
    "strat,strat_class,err",
    [
        (None, None, True),
        ("complex", ComplexOnlyStrategy, False),
        ("concat", ConcatStrategy, False),
        ("delta", DeltaStrategy, False),
    ],
)
def test_strategy_e3nn(strat, strat_class, err):
    if err:
        with pytest.raises(pydantic.ValidationError):
            _ = E3NNModelConfig(strategy=strat)
        return

    model = E3NNModelConfig(strategy=strat).build()
    assert isinstance(model.strategy, strat_class)


def test_model_weights_e3nn():
    model1 = E3NNModelConfig().build()
    model2 = E3NNModelConfig(model_weights=model1.state_dict()).build()

    test_model_params = dict(model2.named_parameters())
    for n, ref_param in model1.named_parameters():
        assert (ref_param == test_model_params[n]).all()


def test_str_irreps_dict_e3nn():
    irreps_str = "0:10,1:5"
    model_config = E3NNModelConfig(irreps_hidden=irreps_str)
    assert model_config.irreps_hidden == "10x0o+10x0e+5x1o+5x1e"


def test_bad_str_irreps_dict_e3nn():
    irreps_str = "0,1:5"
    with pytest.raises(ValueError):
        _ = E3NNModelConfig(irreps_hidden=irreps_str)


def test_str_irreps_bad_dict_e3nn():
    irreps_str = "0k:10,1:5"
    with pytest.raises(ValueError):
        _ = E3NNModelConfig(irreps_hidden=irreps_str)


def test_str_irreps_str_e3nn():
    irreps_str = "10x0o+10x0e+5x1o+5x1e"
    model_config = E3NNModelConfig(irreps_hidden=irreps_str)
    assert model_config.irreps_hidden == "10x0o+10x0e+5x1o+5x1e"


def test_bad_str_irreps_str_e3nn():
    irreps_str = "10x0k+10x0e+5x1o+5x1e"
    with pytest.raises(ValueError):
        _ = E3NNModelConfig(irreps_hidden=irreps_str)


def test_random_seed_schnet():
    rand_config = SchNetModelConfig()
    set_config = SchNetModelConfig(rand_seed=10)

    rand_model1 = rand_config.build()
    rand_model2 = rand_config.build()
    set_model1 = set_config.build()
    set_model2 = set_config.build()

    rand_equal = [
        (p1 == p2).all()
        for p1, p2 in zip(rand_model1.parameters(), rand_model2.parameters())
    ]
    assert sum(rand_equal) < len(rand_equal)

    set_equal = [
        (p1 == p2).all()
        for p1, p2 in zip(set_model1.parameters(), set_model2.parameters())
    ]
    assert sum(set_equal) == len(set_equal)


@pytest.mark.parametrize(
    "pred_r,pred_r_class,pred_r_args",
    [
        (None, None, [None, None]),
        ("pic50", PIC50Readout, [None, None]),
        ("pic50", PIC50Readout, [0.375, 9.5]),
        ("pki", PKiReadout, [None, None]),
    ],
)
def test_readout_schnet(pred_r, pred_r_class, pred_r_args):
    model = SchNetModelConfig(
        pred_readout=pred_r,
        pred_substrate=pred_r_args[0],
        pred_km=pred_r_args[1],
    ).build()

    if pred_r is None:
        assert model.readout is None
        return

    assert isinstance(model.readout, pred_r_class)
    if pred_r == "pic50":
        assert model.readout.substrate == pred_r_args[0]
        assert model.readout.Km == pred_r_args[1]


@pytest.mark.parametrize(
    "strat,strat_class,err",
    [
        (None, None, True),
        ("complex", ComplexOnlyStrategy, False),
        ("concat", ConcatStrategy, False),
        ("delta", DeltaStrategy, False),
    ],
)
def test_strategy_schnet(strat, strat_class, err):
    if err:
        with pytest.raises(pydantic.ValidationError):
            _ = SchNetModelConfig(strategy=strat)
        return

    model = SchNetModelConfig(strategy=strat).build()
    assert isinstance(model.strategy, strat_class)


def test_model_weights_schnet():
    model1 = SchNetModelConfig().build()
    model2 = SchNetModelConfig(model_weights=model1.state_dict()).build()

    test_model_params = dict(model2.named_parameters())
    for n, ref_param in model1.named_parameters():
        assert (ref_param == test_model_params[n]).all()


def test_random_seed_visnet():
    rand_config = ViSNetModelConfig()
    set_config = ViSNetModelConfig(rand_seed=10)

    rand_model1 = rand_config.build()
    rand_model2 = rand_config.build()
    set_model1 = set_config.build()
    set_model2 = set_config.build()

    rand_equal = [
        (p1 == p2).all()
        for p1, p2 in zip(rand_model1.parameters(), rand_model2.parameters())
    ]
    assert sum(rand_equal) < len(rand_equal)

    set_equal = [
        (p1 == p2).all()
        for p1, p2 in zip(set_model1.parameters(), set_model2.parameters())
    ]
    assert sum(set_equal) == len(set_equal)


def test_visnet_from_pyg():
    from torch_geometric.nn.models import ViSNet as PyVisNet
    from mtenn.conversion_utils import ViSNet

    model_params = {
        "lmax": 1,
        "vecnorm_type": None,
        "trainable_vecnorm": False,
        "num_heads": 8,
        "num_layers": 6,
        "hidden_channels": 128,
        "num_rbf": 32,
        "trainable_rbf": False,
        "max_z": 100,
        "cutoff": 5.0,
        "max_num_neighbors": 32,
        "vertex": False,
        "reduce_op": "sum",
        "mean": 0.0,
        "std": 1.0,
        "derivative": False,
        "atomref": None,
    }

    pyg_model = PyVisNet(**model_params)
    visnet_model = ViSNet(model=pyg_model)

    params_equal = [
        (p1 == p2).all()
        for p1, p2 in zip(pyg_model.parameters(), visnet_model.parameters())
    ]
    assert sum(params_equal) == len(params_equal)
