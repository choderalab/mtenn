from mtenn.config import GATModelConfig, E3NNModelConfig, SchNetModelConfig


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
