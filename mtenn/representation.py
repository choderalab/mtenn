"""
This module is mostly here for completeness and convenience. The different
``Representation`` implementations will typically be handled in
:py:mod:`mtenn.conversion_utils`, but because the abstract
:py:class:`Representation <mtenn.representation.Representation>` class only subclasses
the torch ``Module`` class, any model will be able to fit anywhere that is typed as a
``Representation`` object.
"""

import abc
import torch


class Representation(torch.nn.Module, abc.ABC):
    """
    Abstract base class for the ``Representation`` block.
    """

    pass
