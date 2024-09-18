"""
Implementations for the ``Readout`` block in a :py:class:`Model
<mtenn.model.Model>` or :py:class:`GroupedModel <mtenn.model.GroupedModel>`.

This class is intended to contain only simple arithmetic, converting a
model-predicted :math:`\mathrm{\Delta G}` value (in implicit kT units) into some
readout value that can be experimentally measured, eg :math:`\mathrm{pIC_{50}}`.
"""

import abc
import torch
from typing import Optional


class Readout(torch.nn.Module, abc.ABC):
    """
    Abstract base class for the ``Readout`` block. Any subclass needs to implement
    the ``forward`` method in order to be used.
    """

    @abc.abstractmethod
    def forward(self, delta_g):
        """
        For any readout class, this function should take the predicted
        :math:`\mathrm{\Delta G}` value as input, and return whatever transformed value
        the class is implementing.
        """
        raise NotImplementedError("Must implement the `forward` method.")

    def __str__(self):
        return repr(self)


class PIC50Readout(Readout):
    """
    Readout implementation to convert :math:`\Delta \mathrm{G}` values to
    :math:`\mathrm{pIC_{50}}` values. This new implementation assumes implicit energy
    units, **WHICH WILL INVALIDATE MODELS TRAINED PRIOR TO v0.3.0**.

    Assuming implicit energy units:

    .. math::

        \Delta \mathrm{G} &= \mathrm{ln}(\mathrm{K_i})

        \mathrm{K_i} &= \mathrm{exp}(\Delta \mathrm{G})

    Using the Cheng-Prusoff equation:

    .. math::

        \mathrm{K_i} &= \mathrm{\\frac{IC_{50}}{1 + [S]/K_m}}

        \mathrm{exp(\Delta G)} &= \\frac{\mathrm{IC_{50}}}{\mathrm{1 + [S]/K_m}}

        \mathrm{IC_{50}} &= \mathrm{exp(\Delta G) (1 + [S]/K_m)}

        \mathrm{pIC_{50}} &= \mathrm{-log10(exp(\Delta G) * (1 + [S]/K_m))}

        \mathrm{pIC_{50}} &= \mathrm{-log10(exp(\Delta G)) - log10(1 + [S]/K_m)}

        \mathrm{pIC_{50}} &= \mathrm{-\\frac{ln(exp(\Delta G))}{ln(10)} -
        log10(1 + [S]/K_m)}

        \mathrm{pIC_{50}} &= \mathrm{-\\frac{\Delta G}{ln(10)} - log10(1 + [S]/K_m)}

    Alternatively, estimating :math:`\mathrm{K_i}` as the :math:`\mathrm{IC_{50}}`
    value:

    .. math::

        \mathrm{K_i} &= \mathrm{IC_{50}}

        \mathrm{IC_{50}} &= \mathrm{exp(\Delta G)}

        \mathrm{pIC_{50}} &= \mathrm{-log10(exp(\Delta G))}

        \mathrm{pIC_{50}} &= \mathrm{-\\frac{ln(exp(\Delta G))}{ln(10)}}

        \mathrm{pIC_{50}} &= \mathrm{-\\frac{\Delta G}{ln(10)}}
    """

    def __init__(self, substrate: Optional[float] = None, Km: Optional[float] = None):
        """
        Initialize conversion with specified substrate concentration and
        :math:`\mathrm{K_m}`. If either is left blank, the :math:`\mathrm{IC_{50}}`
        approximation will be used.

        Parameters
        ----------
        substrate : float, optional
            Substrate concentration for use in the Cheng-Prusoff equation. Assumed to be
            in the same units as :math:`\mathrm{K_m}`
        Km : float, optional
            :math:`\mathrm{K_m}` value for use in the Cheng-Prusoff equation. Assumed to
            be in the same units as substrate
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
        Method to convert a predicted :math:`\Delta \mathrm{G}` value into a
        :math:`\mathrm{pIC_{50}}` value.

        Parameters
        ----------
        delta_g : torch.Tensor
            Input :math:`\Delta \mathrm{G}` value.

        Returns
        -------
        torch.Tensor
            Calculated :math:`\mathrm{pIC_{50}}` value.
        """
        pic50 = -delta_g / torch.log(torch.tensor(10, dtype=delta_g.dtype))
        # Using Cheng-Prusoff
        if self.cp_val:
            pic50 -= torch.log10(torch.tensor(self.cp_val, dtype=delta_g.dtype))

        return pic50


class PKiReadout(Readout):
    """
    Readout implementation to convert :math:`\Delta \mathrm{G}` values to
    :math:`\mathrm{pK_i}` values. This new implementation assumes implicit energy units,
    **WHICH WILL INVALIDATE MODELS TRAINED PRIOR TO v0.3.0**.

    Assuming implicit energy units:

    .. math::

        \mathrm{\Delta G} &= \mathrm{ln(K_i)}

        \mathrm{K_i} &= \mathrm{exp(\Delta G)}

        \mathrm{pK_i} &= \mathrm{-log10(K_i)}

        \mathrm{pK_i} &= \mathrm{-log10(exp(\Delta G))}

        \mathrm{pK_i} &= \\frac{\mathrm{-ln(exp(\Delta G))}}{\mathrm{ln(10)}}

        \mathrm{pK_i} &= \\frac{\mathrm{-\Delta G}}{\mathrm{ln(10)}}
    """

    def __repr__(self):
        return "PKiReadout()"

    def forward(self, delta_g):
        """
        Method to convert a predicted :math:`\Delta \mathrm{G}` value into a
        :math:`\mathrm{pK_i}` value.

        Parameters
        ----------
        delta_g : torch.Tensor
            Input :math:`\Delta \mathrm{G}` value.

        Returns
        -------
        torch.Tensor
            Calculated :math:`\mathrm{pK_i}` value.
        """
        pki = -delta_g / torch.log(torch.tensor(10, dtype=delta_g.dtype))

        return pki
