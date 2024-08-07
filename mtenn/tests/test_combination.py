from copy import deepcopy
import numpy as np
import pytest
import torch

from mtenn.combination import MeanCombination, MaxCombination
from mtenn.conversion_utils.schnet import SchNet


@pytest.fixture()
def models_and_inputs():
    model_test = SchNet(
        hidden_channels=2, num_filters=2, num_interactions=2, num_gaussians=2
    )
    model_ref = deepcopy(model_test)
    model_ref = SchNet.get_model(model_ref, strategy="complex")

    elem_list = torch.randint(11, size=(10,))
    inp_list = [
        {
            "z": elem_list,
            "pos": torch.rand((10, 3)) * 10,
            "lig": torch.ones(10, dtype=bool),
        }
        for _ in range(5)
    ]
    target = torch.rand(1)
    loss_func = torch.nn.MSELoss()

    return model_test, model_ref, inp_list, target, loss_func


def test_mean_combination(models_and_inputs):
    model_test, model_ref, inp_list, target, loss_func = models_and_inputs

    # Ref calc
    pred_list = [model_ref(X)[0] for X in inp_list]
    pred_ref = torch.stack(pred_list).mean(axis=0)
    loss = loss_func(pred_ref, target)
    loss.backward()

    # Finish setting up GroupedModel
    model_test = SchNet.get_model(
        model_test, grouped=True, strategy="complex", combination=MeanCombination()
    )

    # Test GroupedModel
    pred_test, _ = model_test(inp_list)
    loss = loss_func(pred_test, target)
    loss.backward()

    # Compare
    ref_param_dict = dict(model_ref.named_parameters())
    assert all(
        [
            np.allclose(p.grad, ref_param_dict[n].grad, atol=5e-7)
            for n, p in model_test.named_parameters()
        ]
    )


def test_max_combination(models_and_inputs):
    model_test, model_ref, inp_list, target, loss_func = models_and_inputs

    # Ref calc
    pred_list = [model_ref(X)[0] for X in inp_list]
    pred = torch.logsumexp(torch.stack(pred_list), axis=0)
    loss = loss_func(pred, target)
    loss.backward()

    # Finish setting up GroupedModel
    model_test = SchNet.get_model(
        model_test,
        grouped=True,
        strategy="complex",
        combination=MaxCombination(negate_preds=False, pred_scale=1.0),
    )

    # Test GroupedModel
    pred, _ = model_test(inp_list)
    loss = loss_func(pred, target)
    loss.backward()

    # Compare
    ref_param_dict = dict(model_ref.named_parameters())
    assert all(
        [
            np.allclose(p.grad, ref_param_dict[n].grad, atol=5e-7)
            for n, p in model_test.named_parameters()
        ]
    )
