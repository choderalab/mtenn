"""
Implementations for the ``Strategy`` block in a :py:class:`Model
<mtenn.model.Model>` or :py:class:`GroupedModel <mtenn.model.GroupedModel>`.
"""

import abc
from itertools import permutations
import torch


class Strategy(torch.nn.Module, abc.ABC):
    """
    Abstract base class for the ``Strategy`` block. Any subclass needs to implement
    the ``forward`` method in order to be used.
    """

    @abc.abstractmethod
    def forward(self, comp, *parts):
        """
        For any strategy class, this function should take a complex representation and
        (optionally) any number of "part" representations, and return a single
        :math:`\mathrm{\Delta G}` prediction.
        """
        raise NotImplementedError("Must implement the `forward` method.")


class DeltaStrategy(Strategy):
    """
    Simple strategy for subtracting the sum of the individual component energies
    from the complex energy. This ``Strategy`` requires an ``energy_func``
    :math:`\phi: \mathbb{R}^n \\rightarrow \mathbb{R}` that maps from an n-dimensional
    vector representation (output from a ``Representation`` block) to a scalar-value
    energy prediction.

    .. math::

        \mathrm{G} &= \phi (\mathrm{\\boldsymbol{x}})

        \Delta \mathrm{G_{pred}} &= \mathrm{G_{complex}} - \\sum_n \mathrm{G}_n
    """

    def __init__(self, energy_func):
        """
        Store module for predicting an energy from representation.

        Parameters
        ----------
        energy_func : torch.nn.Module
            Some torch module that will predict an energy from an n-dimension vector
            representation of a structure
        """
        super(DeltaStrategy, self).__init__()
        self.energy_func: torch.nn.Module = energy_func

    def forward(self, comp, *parts):
        """
        Make energy predictions for each representation, and then perform the delta
        calculation.

        Parameters
        ----------
        comp : torch.Tensor
            Complex representation that will be passed to ``self.energy_func``
        parts : list[torch.Tensor], optional
            Representations for all individual parts of the complex (eg ligand and
            protein separately) that will be passed to ``self.energy_func``

        Returns
        -------
        torch.Tensor
            Predicted :math:`\Delta G` value
        """
        # Get energy predictions for each representation
        complex_pred = self.energy_func(comp)
        parts_preds = [self.energy_func(p) for p in parts]
        # Replace invalid predictions with 0
        parts_preds = [
            p if len(p.flatten()) > 0 else torch.zeros_like(complex_pred)
            for p in parts_preds
        ]
        # Calculate delta G
        dG_pred = complex_pred - sum(parts_preds)
        return dG_pred


class ConcatStrategy(Strategy):
    """
    Strategy for combining the complex representation and parts representations
    in some learned manner, using sum-pooling to ensure permutation-invariance
    of the parts. For 3 n-dimensional input representations (eg complex, protein-only,
    and ligand-only), this ``Strategy`` acts as a function
    :math:`\phi: \mathbb{R}^{3n} \\rightarrow \mathbb{R}` that predicts a scalar-value
    :math:`\Delta G` prediction.

    The input :math:`\mathrm{\\boldsymbol{x}}` to :math:`\phi` is computed in a
    permutation-invariant manner. For a protein-ligand complex, this looks like:

    .. math::

        \mathrm{\\boldsymbol{x}_{parts}} &= [\mathrm{\\boldsymbol{x}_{protein}},
        \mathrm{\\boldsymbol{x}_{ligand}}] + [\mathrm{\\boldsymbol{x}_{ligand}},
        \mathrm{\\boldsymbol{x}_{protein}}]

        \mathrm{\\boldsymbol{x}} &= [\mathrm{\\boldsymbol{x}_{complex}},
        \mathrm{\\boldsymbol{x}_{parts}}]

        \Delta \mathrm{G_{pred}} &= \phi (\mathrm{\\boldsymbol{x}})

    In general, we will sum every permutation of the non-complex representations, and
    then this sum will be concatenated to the complex representation.

    In its current iteration, this ``Strategy`` does not require you to specify the
    dimensionality (:math:`n`) of each representation. Instead, the first time an
    instance of this ``Strategy`` is used, it will calculate the required input size and
    initialize a one-layer linear network of the appropriate dimensionality.
    """

    def __init__(self, input_size, extract_key=None, layer_norm=False):
        """
        Set the key to use to access vector representations if ``dict`` s are passed to
        the ``forward`` call.

        Parameters
        ----------
        input_size : int
            Input size of linear model
        extract_key : str, optional
            Key to use to extract representation from a dict
        layer_norm: bool, default=False
            Apply a ``LayerNorm`` normalization before passing through the linear layer

        """
        super(ConcatStrategy, self).__init__()
        if layer_norm:
            self.reduce_nn = torch.nn.Sequential(
                torch.nn.LayerNorm(input_size), torch.nn.Linear(input_size, 1)
            )
        else:
            self.reduce_nn = torch.nn.Linear(input_size, 1)
        self.extract_key = extract_key

    def forward(self, comp, *parts):
        """
        Calculate permutation-invariant concatenation of all representations, and pass
        through a one-layer linear NN. This network will be initialized based on the
        input sizes the first time this method is called for a given instance of this
        class.

        Parameters
        ----------
        comp : torch.Tensor
            Complex representation
        parts : list[torch.Tensor], optional
            Representations for all individual parts of the complex (eg ligand and
            protein separately)

        Returns
        -------
        torch.Tensor
            Predicted :math:`\Delta G` value
        """
        # Extract representation from dict
        if self.extract_key and isinstance(comp, dict):
            comp = comp[self.extract_key]
            parts = [p[self.extract_key] for p in parts]

        # Flatten tensors
        comp = comp.flatten()
        parts = [p.flatten() for p in parts]

        # Enumerate all possible permutations of parts and add together
        parts_size = sum([len(p) for p in parts])
        parts_cat = torch.zeros((parts_size), device=comp.device)
        for idxs in permutations(range(len(parts)), len(parts)):
            parts_cat += torch.cat([parts[i] for i in idxs])

        # Concat comp w/ permut-invariant parts representation
        full_embedded = torch.cat([comp, parts_cat])

        return self.reduce_nn(full_embedded)


class ComplexOnlyStrategy(Strategy):
    """
    Strategy to only predict based on the complex representation. This is useful if you
    want to make a prediction on just the ligand or just the protein, and essentially
    just reduces to a standard version of whatever your underlying model is.
    """

    def __init__(self, energy_func):
        """
        Store module for predicting an energy from representation.

        Parameters
        ----------
        energy_func : torch.nn.Module
            Some torch module that will predict an energy from an n-dimension vector
            representation of a structure
        """
        super().__init__()
        self.energy_func: torch.nn.Module = energy_func

    def forward(self, comp, *parts):
        """
        Make energy prediction for the complex representation.

        Parameters
        ----------
        comp : torch.Tensor
            Complex representation that will be passed to ``self.energy_func``
        parts : list[torch.Tensor], optional
            Ignored, but present just to match the signatures

        Returns
        -------
        torch.Tensor
            Predicted value
        """
        complex_pred = self.energy_func(comp)
        return complex_pred
