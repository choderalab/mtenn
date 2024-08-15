import pytest
import torch

from mtenn.conversion_utils.gat import GAT
import rdkit.Chem as Chem
from torch_geometric.nn.models import GAT as PygGAT


@pytest.fixture
def model_input():
    # Build mol
    smiles = "CCCC"
    mol = Chem.MolFromSmiles(smiles)

    # Get atomic numbers and bond indices (both directions)
    atomic_nums = [a.GetAtomicNum() for a in mol.GetAtoms()]
    bond_idxs = [
        atom_pair
        for bond in mol.GetBonds()
        for atom_pair in (
            (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
            (bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()),
        )
    ]
    # Add self bonds
    bond_idxs += [(a.GetIdx(), a.GetIdx()) for a in mol.GetAtoms()]

    # Encode atomic numbers as one-hot, assume max num of 100
    feature_tensor = torch.nn.functional.one_hot(
        torch.tensor(atomic_nums), num_classes=100
    ).to(dtype=torch.float)
    # Format bonds in correct shape
    bond_list_tensor = torch.tensor(bond_idxs).t()

    return {"x": feature_tensor, "edge_index": bond_list_tensor, "smiles": smiles}


def test_build_gat_directly_kwargs():
    model = GAT(in_channels=-1, hidden_channels=32, num_layers=2)
    assert model.gnn.num_layers == 2

    assert model.gnn.convs[0].in_channels == -1
    assert model.gnn.convs[0].out_channels == 32

    assert model.gnn.convs[1].in_channels == 32
    assert model.gnn.convs[1].out_channels == 32


def test_build_gat_from_pyg_gat():
    pyg_model = PygGAT(in_channels=10, hidden_channels=32, num_layers=2)
    model = GAT(model=pyg_model)

    # Check set up as before
    assert model.gnn.num_layers == 2

    assert model.gnn.convs[0].in_channels == 10
    assert model.gnn.convs[0].out_channels == 32

    assert model.gnn.convs[1].in_channels == 32
    assert model.gnn.convs[1].out_channels == 32

    # Check that model weights got copied
    ref_params = dict(pyg_model.state_dict())
    for n, model_param in model.gnn.named_parameters():
        assert (model_param == ref_params[n]).all()


def test_gat_can_predict(model_input):
    model = GAT(in_channels=-1, hidden_channels=32, num_layers=2)
    _ = model(model_input)


def test_representation_is_correct():
    model = GAT(in_channels=10, hidden_channels=32, num_layers=2)
    rep = model._get_representation()

    model_params = dict(model.gnn.named_parameters())
    for n, rep_param in rep.named_parameters():
        assert torch.allclose(rep_param, model_params[n])


def test_get_model_no_ref():
    model = GAT.get_model(in_channels=10, hidden_channels=32, num_layers=2)

    assert isinstance(model.representation, GAT)
    assert model.readout is None


def test_get_model_ref():
    ref_model = GAT(in_channels=10, hidden_channels=32, num_layers=2)
    model = GAT.get_model(model=ref_model)

    assert isinstance(model.representation, GAT)
    assert model.readout is None

    ref_params = dict(ref_model.named_parameters())
    for n, model_param in model.representation.named_parameters():
        assert torch.allclose(model_param, ref_params[n])
