import pydantic
import pytest

from mtenn.config import (
    ModelConfig,
    LigandOnlyModelConfig,
    GATRepresentationConfig,
    E3NNRepresentationConfig,
    SchNetRepresentationConfig,
)
from mtenn.readout import PIC50Readout, PKiReadout
from mtenn.strategy import ComplexOnlyStrategy, ConcatStrategy, DeltaStrategy


def test_random_seed_gat():
    rand_config = LigandOnlyModelConfig(representation=GATRepresentationConfig())
    set_config = LigandOnlyModelConfig(
        representation=GATRepresentationConfig(), rand_seed=10
    )

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
    model = LigandOnlyModelConfig(
        representation=GATRepresentationConfig(),
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
    model1 = LigandOnlyModelConfig(representation=GATRepresentationConfig()).build()
    model2 = LigandOnlyModelConfig(
        representation=GATRepresentationConfig(), model_weights=model1.state_dict()
    ).build()

    test_model_params = dict(model2.named_parameters())
    for n, ref_param in model1.named_parameters():
        assert (ref_param == test_model_params[n]).all()


def test_random_seed_e3nn():
    rand_config = ModelConfig(representation=E3NNRepresentationConfig())
    set_config = ModelConfig(representation=E3NNRepresentationConfig(), rand_seed=10)

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
    model = ModelConfig(
        representation=E3NNRepresentationConfig(),
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
            _ = ModelConfig(representation=E3NNRepresentationConfig(), strategy=strat)
        return

    model = ModelConfig(
        representation=E3NNRepresentationConfig(), strategy=strat
    ).build()
    assert isinstance(model.strategy, strat_class)


def test_model_weights_e3nn():
    model1 = ModelConfig(representation=E3NNRepresentationConfig()).build()
    model2 = ModelConfig(
        representation=E3NNRepresentationConfig(), model_weights=model1.state_dict()
    ).build()

    test_model_params = dict(model2.named_parameters())
    for n, ref_param in model1.named_parameters():
        assert (ref_param == test_model_params[n]).all()


def test_random_seed_schnet():
    rand_config = ModelConfig(representation=SchNetRepresentationConfig())
    set_config = ModelConfig(representation=SchNetRepresentationConfig(), rand_seed=10)

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
    model = ModelConfig(
        representation=SchNetRepresentationConfig(),
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
            _ = ModelConfig(representation=SchNetRepresentationConfig(), strategy=strat)
        return

    model = ModelConfig(
        representation=SchNetRepresentationConfig(), strategy=strat
    ).build()
    assert isinstance(model.strategy, strat_class)


def test_model_weights_schnet():
    model1 = ModelConfig(representation=SchNetRepresentationConfig()).build()
    model2 = ModelConfig(
        representation=SchNetRepresentationConfig(), model_weights=model1.state_dict()
    ).build()

    test_model_params = dict(model2.named_parameters())
    for n, ref_param in model1.named_parameters():
        assert (ref_param == test_model_params[n]).all()
