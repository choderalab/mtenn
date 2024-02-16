import pytest
import torch

from mtenn.model import GroupedModel, LigandOnlyModel, Model
from mtenn.representation import Representation
from mtenn.strategy import Strategy
from mtenn.readout import Readout


@pytest.fixture
def toy_model_setup():
    """
    Simple modules that don't do much so we can check that the right stuff is
    happening internally.
    """

    class ToyRepresentation(Representation):
        """
        Wrapper for torch.nn.Identity so we can handle dict unpacking.
        """

        def __init__(self):
            super().__init__()

        def forward(self, data):
            return torch.nn.Identity()(data["x"])

    class ToyStrategy(Strategy):
        """
        Just take the sum of each representation.
        """

        def __init__(self):
            super().__init__()

        def forward(self, comp, *parts):
            return comp.sum() - sum([p.sum() for p in parts])

    class ToyReadout(Readout):
        """
        Toy Readout for testing. Just add 5 since results before this should be 0.
        """

        def __init__(self):
            super().__init__()

        def forward(self, val):
            return val + 5

    return {
        "representation": ToyRepresentation(),
        "strategy": ToyStrategy(),
        "readout": ToyReadout(),
    }


@pytest.fixture
def toy_grouped_model_setup(toy_model_setup):
    """
    Simple modules that don't do much so we can check that the right stuff is
    happening internally.
    """

    # Not actually subclassing Combination here bc the real thing is a bit more complex
    #  than what we need for testing the GroupedModel class
    class ToyCombination(torch.nn.Module):
        """
        Just take the mean of all inputs.
        """

        def __init__(self):
            super().__init__()

        def forward(self, preds):
            return torch.stack(preds).mean(axis=None)

    toy_model_setup["combination"] = ToyCombination()

    toy_model_setup["pred_readout"] = toy_model_setup["readout"]
    toy_model_setup["comb_readout"] = toy_model_setup["readout"]
    del toy_model_setup["readout"]

    return toy_model_setup


@pytest.fixture
def toy_inputs():
    """
    Small toy inputs so we can test the different functionalities of the Model classes.
    The actual testing for the real Representation/Strategy/Readouts should all be done
    in their respective test files.
    """

    # Match dtype with defaults for torch models
    complex_pose = {
        "x": torch.arange(10, dtype=torch.float32),
        "lig": torch.tensor([False] * 7 + [True] * 3),
        "target": 0.0,
    }
    prot_pose = {
        "x": torch.arange(7, dtype=torch.float32),
        "lig": torch.tensor([False] * 7),
        "target": 0.0,
    }
    lig_pose = {
        "x": torch.arange(7, 10, dtype=torch.float32),
        "lig": torch.tensor([True] * 3),
        "target": 0.0,
    }

    return complex_pose, prot_pose, lig_pose


def test_model_split_part(toy_inputs):
    complex_pose, prot_pose, lig_pose = toy_inputs

    prot_split, lig_split = Model._split_parts(complex_pose)
    assert (prot_pose["x"] == prot_split["x"]).all()
    assert (lig_pose["x"] == lig_split["x"]).all()


def test_cant_split_without_lig():
    with pytest.raises(RuntimeError):
        Model._split_parts({})


def test_model_building(toy_model_setup):
    _ = Model(**toy_model_setup)


def test_model_get_representation(toy_model_setup, toy_inputs):
    model = Model(**toy_model_setup)
    for inp in toy_inputs:
        assert (model.get_representation(inp) == inp["x"]).all()


def test_model_forward_explicit_parts(toy_model_setup, toy_inputs):
    model = Model(**toy_model_setup)

    pred = model(*toy_inputs)  # complex, protein, lig
    target = toy_inputs[0]["x"].sum() - (
        toy_inputs[1]["x"].sum() + toy_inputs[2]["x"].sum()
    )

    assert pred[0] == target + 5
    assert pred[1] == [target]


def test_model_forward_auto_splitting(toy_model_setup, toy_inputs):
    model = Model(**toy_model_setup)

    pred = model(toy_inputs[0])  # just complex
    target = toy_inputs[0]["x"].sum() - (
        toy_inputs[1]["x"].sum() + toy_inputs[2]["x"].sum()
    )

    assert pred[0] == target + 5
    assert pred[1] == [target]  # this is pre-Readout


def test_model_forward_no_readout(toy_model_setup, toy_inputs):
    model = Model(
        representation=toy_model_setup["representation"],
        strategy=toy_model_setup["strategy"],
    )

    pred = model(toy_inputs[0])  # just complex
    target = toy_inputs[0]["x"].sum() - (
        toy_inputs[1]["x"].sum() + toy_inputs[2]["x"].sum()
    )

    assert pred[0] == target
    assert pred[1] == [target]


def test_grouped_model_building(toy_grouped_model_setup):
    model = GroupedModel(**toy_grouped_model_setup)

    assert model.readout == toy_grouped_model_setup["pred_readout"]


# This will take some reworking bc the GroupedModel calls `backward` internally, so the
#  parts all need to be differentiable
@pytest.mark.xfail
def test_grouped_model_forward(toy_grouped_model_setup, toy_inputs):
    model = GroupedModel(**toy_grouped_model_setup)

    # Pass list of the same thing 3 times
    pred = model([toy_inputs[0]] * 3)

    # Each individual pose should return this, which should also then be the mean
    target = (
        toy_inputs[0]["x"].sum()
        - (toy_inputs[1]["x"].sum() + toy_inputs[2]["x"].sum())
        + 5
    )

    assert pred[0] == target + 5
    assert pred[1] == [target] * 3
