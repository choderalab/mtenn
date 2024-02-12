import pytest
import torch

from mtenn.readout import PIC50Readout, PKiReadout


@pytest.fixture
def values():
    cp_values = [0.375, 9.5]

    # Input dG value
    dG = torch.tensor(-5, dtype=float)
    # No Cheng-Prusoff equation
    raw_pic50 = -dG / torch.log(torch.tensor(10))
    # Using Cheng-Prusoff
    cp_pic50 = raw_pic50 - torch.log10(
        torch.tensor(1 + cp_values[0] / cp_values[1], dtype=raw_pic50.dtype)
    )

    return cp_values, dG, raw_pic50, cp_pic50


def test_pic50_readout_repr():
    r = PIC50Readout(substrate=10, Km=10)
    assert repr(r) == "PIC50Readout(substrate=10, Km=10)"


def test_pic50_readout_str():
    r = PIC50Readout(substrate=10, Km=10)
    assert str(r) == "PIC50Readout(substrate=10, Km=10)"


def test_pic50_readout_no_cheng_prusoff(values):
    r = PIC50Readout()
    _, dG, raw_pic50, _ = values

    assert r(dG) == raw_pic50


def test_pic50_readout_cheng_prusoff(values):
    cp_values, dG, _, cp_pic50 = values
    r = PIC50Readout(*cp_values)

    assert torch.isclose(r(dG), cp_pic50)


def test_pki_readout_repr():
    r = PKiReadout()
    assert repr(r) == "PKiReadout()"


def test_pki_readout_str():
    r = PKiReadout()
    assert str(r) == "PKiReadout()"


def test_pki_readout(values):
    r = PKiReadout()
    dG = values[1]

    assert torch.isclose(r(dG), -torch.log10(torch.exp(dG)))
