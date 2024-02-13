from mtenn.config import GATModelConfig, E3NNModelConfig, SchNetModelConfig, ViSNetModelConfig
from mtenn.conversion_utils.visnet import HAS_VISNET
import pytest

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

@pytest.mark.skipif(not HAS_VISNET, reason="requires VisNet from nightly PyG")
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

@pytest.mark.skipif(not HAS_VISNET, reason="requires VisNet from nightly PyG")
def test_visnet_from_pyg():
    from torch_geometric.nn.models import ViSNet as PyVisNet
    from mtenn.conversion_utils import ViSNet
    model_params={
        'lmax': 1,
        'vecnorm_type': None,
        'trainable_vecnorm': False,
        'num_heads': 8,
        'num_layers': 6,
        'hidden_channels': 128,
        'num_rbf': 32,
        'trainable_rbf': False,
        'max_z': 100,
        'cutoff': 5.0,
        'max_num_neighbors': 32,
        'vertex': False,
        'reduce_op': "sum",
        'mean': 0.0,
        'std': 1.0,
        'derivative': False,
        'atomref': None,
    }

    pyg_model = PyVisNet(**model_params)    
    visnet_model = ViSNet(model=pyg_model)

    params_equal = [
        (p1 == p2).all()
        for p1, p2 in zip(pyg_model.parameters(), visnet_model.parameters())
    ]
    assert sum(params_equal) == len(params_equal)
