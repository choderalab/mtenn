import pytest

from mtenn.conversion_utils.schnet import SchNet, PygSchNet
from torch_geometric.nn import SumAggregation


@pytest.fixture
def schnet_kwargs():
    return {
        "hidden_channels": 16,
        "num_filters": 16,
        "num_interactions": 2,
        "num_gaussians": 5,
        "interaction_graph": None,
        "cutoff": 10,
        "max_num_neighbors": 16,
        "readout": "add",
        "dipole": False,
        "mean": None,
        "std": None,
        "atomref": None,
    }


def test_build_schnet_directly_kwargs(schnet_kwargs):
    model = SchNet(**schnet_kwargs)

    # Directly stored parameters
    assert model.hidden_channels == schnet_kwargs["hidden_channels"]
    assert model.num_filters == schnet_kwargs["num_filters"]
    assert model.num_interactions == schnet_kwargs["num_interactions"]
    assert model.num_gaussians == schnet_kwargs["num_gaussians"]
    assert model.cutoff == schnet_kwargs["cutoff"]
    assert model.dipole == schnet_kwargs["dipole"]
    assert isinstance(model.readout, SumAggregation)
    assert model.mean == schnet_kwargs["mean"]
    assert model.std == schnet_kwargs["std"]
    assert model.atomref is None

    # Indirect ones
    interaction_graph = model.interaction_graph
    assert interaction_graph.cutoff == schnet_kwargs["cutoff"]
    assert interaction_graph.max_num_neighbors == schnet_kwargs["max_num_neighbors"]


def test_build_schnet_from_pygschnet(schnet_kwargs):
    ref_model = PygSchNet(**schnet_kwargs)
    model = SchNet(model=ref_model)

    # Directly stored parameters
    assert model.hidden_channels == ref_model.hidden_channels
    assert model.num_filters == ref_model.num_filters
    assert model.num_interactions == ref_model.num_interactions
    assert model.num_gaussians == ref_model.num_gaussians
    assert model.cutoff == ref_model.cutoff
    assert model.dipole == ref_model.dipole
    assert model.readout == ref_model.readout
    assert model.mean == ref_model.mean
    assert model.std == ref_model.std
    assert model.atomref == ref_model.atomref

    # Indirect ones
    interaction_graph = model.interaction_graph
    ref_interaction_graph = ref_model.interaction_graph
    assert interaction_graph.cutoff == ref_interaction_graph.cutoff
    assert (
        interaction_graph.max_num_neighbors == ref_interaction_graph.max_num_neighbors
    )
