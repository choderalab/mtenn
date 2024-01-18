import abc
import torch
from typing import Optional


class Readout(torch.nn.Module, abc.ABC):
    def __str__(self):
        return repr(self)


class PIC50Readout(Readout):
    """
    Readout implementation to convert delta G values to pIC50 values. This new
    implementation assumes implicit energy units, WHICH WILL INVALIDATE MODELS TRAINED
    PRIOR TO v0.3.0.
    Assuming implicit energy units:
        deltaG = ln(Ki)
        Ki = exp(deltaG)
    Using the Cheng-Prusoff equation:
        Ki = IC50 / (1 + [S]/Km)
        exp(deltaG) = IC50 / (1 + [S]/Km)
        IC50 = exp(deltaG) * (1 + [S]/Km)
        pIC50 = -log10(exp(deltaG) * (1 + [S]/Km))
        pIC50 = -log10(exp(deltaG)) - log10(1 + [S]/Km)
        pIC50 = -ln(exp(deltaG))/ln(10) - log10(1 + [S]/Km)
        pIC50 = -deltaG/ln(10) - log10(1 + [S]/Km)
    Estimating Ki as the IC50 value:
        Ki = IC50
        IC50 = exp(deltaG)
        pIC50 = -log10(exp(deltaG))
        pIC50 = -ln(exp(deltaG))/ln(10)
        pIC50 = -deltaG/ln(10)
    """

    def __init__(self, substrate: Optional[float] = None, Km: Optional[float] = None):
        """
        Initialize conversion with specified substrate concentration and Km. If either
        is left blank, the IC50 approximation will be used.

        Parameters
        ----------
        substrate : float, optional
            Substrate concentration for use in the Cheng-Prusoff equation. Assumed to be
            in the same units as Km
        Km : float, optional
            Km value for use in the Cheng-Prusoff equation. Assumed to be in the same
            units as substrate
        """
        super(PIC50Readout, self).__init__()

        self.substrate = substrate
        self.Km = Km

        if substrate and Km:
            self.cp_val = 1 + substrate / Km
        else:
            self.cp_val = None

    def __repr__(self):
        return f"PIC50Readout(substrate={self.substrate}, Km={self.Km})"

    def forward(self, delta_g):
        """
        Method to convert a predicted delta G value into a pIC50 value.

        Parameters
        ----------
        delta_g : torch.Tensor
            Input delta G value.

        Returns
        -------
        float
            Calculated pIC50 value.
        """
        pic50 = -delta_g / torch.log(torch.tensor(10, dtype=delta_g.dtype))
        # Using Cheng-Prusoff
        if self.cp_val:
            pic50 -= torch.log10(torch.tensor(self.cp_val, dtype=delta_g.dtype))

        return pic50



class KiReadout(Readout):
    """
    Readout implementation to convert delta G values to Ki values. This new
    implementation assumes implicit energy units, WHICH WILL INVALIDATE MODELS TRAINED
    PRIOR TO v0.3.0.
    Assuming implicit energy units:
        deltaG = ln(Ki)
        Ki = exp(deltaG)
    """

    def __init__(self):
        """
        Initialization.

        Parameters
        ----------
        None
        """
        super(KiReadout, self).__init__()

    def __repr__(self):
        return f"KiReadout()"

    def forward(self, delta_g):
        """
        Method to convert a predicted delta G value into a Ki value.

        Parameters
        ----------
        delta_g : torch.Tensor
            Input delta G value.

        Returns
        -------
        float
            Calculated Ki value.
        """
        ki = torch.exp(delta_g)

        return ki
