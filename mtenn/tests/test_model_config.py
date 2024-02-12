import pytest

from mtenn.config import GATModelConfig, E3NNModelConfig, SchNetModelConfig
from mtenn.readout import PIC50Readout, PKiReadout


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
