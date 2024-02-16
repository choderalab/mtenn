import pytest

from e3nn.nn.models.gate_points_2101 import Network
from e3nn.o3 import Irreps
from mtenn.conversion_utils.e3nn import E3NN


@pytest.fixture
def e3nn_kwargs():
    return {
        "irreps_in": "5x0e+2x1o",
        "irreps_hidden": "10x0e+10x0o+1o+1e",
        "irreps_out": "0e",
        "irreps_node_attr": "0e",
        "irreps_edge_attr": Irreps.spherical_harmonics(2),
        "layers": 5,
        "max_radius": 10,
        "number_of_basis": 5,
        "radial_layers": 5,
        "radial_neurons": 32,
        "num_neighbors": 10,
        "num_nodes": 100,
        "reduce_output": True,
    }


def test_build_e3nn_directly_kwargs(e3nn_kwargs):
    model = E3NN(**e3nn_kwargs)

    # Directly stored parameters
    assert model.irreps_in == Irreps(e3nn_kwargs["irreps_in"])
    assert model.irreps_hidden == Irreps(e3nn_kwargs["irreps_hidden"])
    assert model.irreps_out == Irreps(e3nn_kwargs["irreps_out"])
    assert model.irreps_node_attr == Irreps(e3nn_kwargs["irreps_node_attr"])
    assert model.irreps_edge_attr == Irreps(e3nn_kwargs["irreps_edge_attr"])
    assert len(model.layers) == e3nn_kwargs["layers"] + 1
    assert model.max_radius == e3nn_kwargs["max_radius"]
    assert model.number_of_basis == e3nn_kwargs["number_of_basis"]
    assert model.num_nodes == e3nn_kwargs["num_nodes"]
    assert model.reduce_output == e3nn_kwargs["reduce_output"]

    # Indirect ones
    conv = model.layers[-1]
    assert len(conv.fc.hs) - 2 == e3nn_kwargs["radial_layers"]
    assert conv.fc.hs[1] == e3nn_kwargs["radial_neurons"]
    assert conv.num_neighbors == e3nn_kwargs["num_neighbors"]


def test_build_e3nn_from_e3nn_network(e3nn_kwargs):
    ref_model = Network(**e3nn_kwargs)
    model = E3NN(model=ref_model)

    # Directly stored parameters
    assert model.irreps_in == ref_model.irreps_in
    assert model.irreps_hidden == ref_model.irreps_hidden
    assert model.irreps_out == ref_model.irreps_out
    assert model.irreps_node_attr == ref_model.irreps_node_attr
    assert model.irreps_edge_attr == ref_model.irreps_edge_attr
    assert len(model.layers) == len(ref_model.layers)
    assert model.max_radius == ref_model.max_radius
    assert model.number_of_basis == ref_model.number_of_basis
    assert model.num_nodes == ref_model.num_nodes
    assert model.reduce_output == ref_model.reduce_output

    # Indirect ones
    ref_conv = ref_model.layers[-1]
    conv = model.layers[-1]
    assert len(conv.fc.hs) == len(ref_conv.fc.hs)
    assert conv.fc.hs[1] == ref_conv.fc.hs[1]
    assert conv.num_neighbors == ref_conv.num_neighbors
