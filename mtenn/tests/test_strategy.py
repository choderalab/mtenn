import pytest
import torch

from mtenn.strategy import ComplexOnlyStrategy, ConcatStrategy, DeltaStrategy


@pytest.fixture
def energy_func():
    """
    Simple linear model with all weights == 1 for easy testing.
    """
    model = torch.nn.Linear(5, 1, bias=False)
    model.load_state_dict({"weight": torch.ones_like(model.weight)})

    return model


@pytest.fixture
def reduce_nn():
    """
    Simple linear model with all weights == 1 for easy testing.
    """
    model = torch.nn.Linear(15, 1, bias=False)
    model.load_state_dict({"weight": torch.ones_like(model.weight)})

    return model


@pytest.fixture
def inputs():
    """
    Complex, protein, and ligand reps.
    """
    comp = torch.arange(5, dtype=torch.float32)
    prot = torch.arange(5, 10, dtype=torch.float32)
    lig = torch.arange(10, 15, dtype=torch.float32)

    return comp, prot, lig


def test_complex_only(energy_func, inputs):
    strat = ComplexOnlyStrategy(energy_func)
    assert strat(*inputs) == 10  # sum(0, 1, 2, 3, 4)


def test_concat_strat_no_extract_key(reduce_nn, inputs):
    strat = ConcatStrategy()
    strat.reduce_nn = reduce_nn

    comp, prot, lig = inputs

    # What should be going on inside the concat strat
    full_rep = torch.cat([prot, lig]) + torch.cat([lig, prot])
    full_rep = torch.cat([comp, full_rep])
    ref_sum = full_rep.sum(axis=None)

    # Test
    pred = strat(comp, prot, lig)
    assert pred == ref_sum
