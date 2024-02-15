import pytest
import torch

from mtenn.model import Model
from mtenn.readout import Readout


@pytest.fixture
def toy_model_setup():
    """
    Simple linear models with weights all 1 so we can check that the right stuff is
    happening internally.
    """

    class LinWrapper(torch.nn.Linear):
        """
        Wrapper for torch.nn.Linear so we can handle dict unpacking.
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            wts_dict = {"weight": torch.ones_like(self.weight)}
            if self.bias is not None:
                wts_dict["bias"] = torch.ones_like(self.bias)

            self.load_state_dict(wts_dict)

        def forward(self, data):
            return super().forward(data["x"])

    class ToyReadout(Readout):
        """
        Toy Readout for testing. Just multiply by 5.
        """

        def __init__(self):
            super().__init__()

        def forward(self, val):
            return val * 5

    return {
        "representation": LinWrapper(10, 10, bias=False),
        "strategy": LinWrapper(10, 1, bias=False),
        "readout": ToyReadout(),
    }


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
    }
    prot_pose = {
        "x": torch.arange(7, dtype=torch.float32),
        "lig": torch.tensor([False] * 7),
    }
    lig_pose = {
        "x": torch.arange(7, 10, dtype=torch.float32),
        "lig": torch.tensor([True] * 3),
    }

    return complex_pose, prot_pose, lig_pose


def test_model_split_part(toy_inputs):
    complex_pose, prot_pose, lig_pose = toy_inputs

    prot_split, lig_split = Model._split_parts(complex_pose)
    assert (prot_pose["x"] == prot_split["x"]).all()
    assert (lig_pose["x"] == lig_split["x"]).all()
