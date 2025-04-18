import pytest
import torch

from mtenn.strategy import (
    ComplexOnlyStrategy,
    ConcatStrategy,
    DeltaStrategy,
    SplitDeltaStrategy,
)


@pytest.fixture
def energy_func():
    """
    Simple linear model with all weights == 1 for easy testing.
    """
    model = torch.nn.Linear(5, 1, bias=False)
    model.load_state_dict({"weight": torch.ones_like(model.weight)})

    return model


@pytest.fixture
def protein_energy_func():
    """
    Simple linear model with all weights == 1 for easy testing.
    """
    model = torch.nn.Linear(5, 1, bias=False)
    model.load_state_dict({"weight": torch.ones_like(model.weight) * 2})

    return model


@pytest.fixture
def ligand_energy_func():
    """
    Simple linear model with all weights == 1 for easy testing.
    """
    model = torch.nn.Linear(5, 1, bias=False)
    model.load_state_dict({"weight": torch.ones_like(model.weight) * 3})

    return model


@pytest.fixture
def reduce_nn_wts_dict():
    """
    Simple linear model with all weights == 1 for easy testing.
    """
    return {"weight": torch.ones((1, 15)), "bias": torch.ones((1,))}


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


def test_concat_strat_no_extract_key(reduce_nn_wts_dict, inputs):
    comp, prot, lig = inputs

    input_size = len(comp.flatten()) + len(prot.flatten()) + len(lig.flatten())
    strat = ConcatStrategy(input_size=input_size)
    strat.reduce_nn.load_state_dict(reduce_nn_wts_dict)

    # What should be going on inside the concat strat
    full_rep = torch.cat([prot, lig]) + torch.cat([lig, prot])
    full_rep = torch.cat([comp, full_rep])
    ref_sum = full_rep.sum(axis=None) + 1

    # Test
    pred = strat(comp, prot, lig)
    assert pred == ref_sum


def test_concat_strat_extract_key(reduce_nn_wts_dict, inputs):
    comp, prot, lig = inputs

    input_size = len(comp.flatten()) + len(prot.flatten()) + len(lig.flatten())
    strat = ConcatStrategy(input_size=input_size, extract_key="x")
    strat.reduce_nn.load_state_dict(reduce_nn_wts_dict)

    # What should be going on inside the concat strat
    full_rep = torch.cat([prot, lig]) + torch.cat([lig, prot])
    full_rep = torch.cat([comp, full_rep])
    ref_sum = full_rep.sum(axis=None) + 1

    # Test
    pred = strat({"x": comp}, {"x": prot}, {"x": lig})
    assert pred == ref_sum


def test_delta_strat(energy_func, inputs):
    strat = DeltaStrategy(energy_func=energy_func)
    comp, prot, lig = inputs

    ref_value = comp.sum(axis=None) - (prot.sum(axis=None) + lig.sum(axis=None))
    pred = strat(*inputs)

    assert pred == ref_value


@pytest.mark.parametrize(
    "use_ligand_energy_func,use_protein_energy_func",
    [(None, None), (True, None), (None, True), (True, True)],
)
def test_split_delta_strat(
    inputs,
    energy_func,
    protein_energy_func,
    ligand_energy_func,
    use_ligand_energy_func,
    use_protein_energy_func,
):
    comp, prot, lig = inputs

    ref_value = comp.sum(axis=None)
    if use_ligand_energy_func:
        use_ligand_energy_func = ligand_energy_func
        ref_value -= (3 * lig).sum(axis=None)
    else:
        ref_value -= lig.sum(axis=None)
    if use_protein_energy_func:
        use_protein_energy_func = protein_energy_func
        ref_value -= (2 * prot).sum(axis=None)
    else:
        ref_value -= prot.sum(axis=None)

    strat = SplitDeltaStrategy(
        complex_energy_func=energy_func,
        ligand_energy_func=use_ligand_energy_func,
        protein_energy_func=use_protein_energy_func,
    )
    pred = strat(comp=comp, prot=prot, lig=lig)

    assert pred == ref_value


def test_split_delta_strat_needs_complex_energy_func():
    with pytest.raises(ValueError):
        SplitDeltaStrategy(complex_energy_func=None)
