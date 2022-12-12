from copy import deepcopy
from itertools import permutations
import torch


class Model(torch.nn.Module):
    """
    Model object containing a `representation` Module that will take an input
    and convert it into some representation, and a `strategy` module that will
    take a complex representation and any number of constituent "part"
    representations, and convert to a final scalar value.
    """

    def __init__(self, representation, strategy, readout=None):
        super(Model, self).__init__()
        self.representation: Representation = representation
        self.strategy: Strategy = strategy
        self.readout: Readout = readout

    def get_representation(self, *args, **kwargs):
        """
        Takes system topolgy and coordinates and returns Nxhidden dimension
        representation.

        Parameters
        ----------

        Returns
        -------
        """

        return self.representation(*args, **kwargs)

    def forward(self, comp, *parts):
        ## This implementation of the forward function assumes the
        ##  get_representation function takes a single data object
        complex_rep = self.get_representation(comp)

        if len(parts) == 0:
            parts = Model._split_parts(comp)
        parts_rep = [self.get_representation(p) for p in parts]

        energy_val = self.strategy(complex_rep, *parts_rep)
        if self.readout:
            return self.readout(energy_val)
        else:
            return energy_val

    @staticmethod
    def _split_parts(comp):
        """
        Helper method to split up the complex representation into different
        parts for protein and ligand.

        Parameters
        ----------
        comp : Dict[str, object]
            Dictionary representing the complex data object. Must have "lig" as
            a key that contains the index for splitting the data.

        Returns
        -------
        Dict[str, object]
            Protein representation
        Dict[str, object]
            Ligand representation
        """
        try:
            idx = comp["lig"]
        except KeyError:
            raise RuntimeError('Data object has no key "lig".')

        prot_rep = {}
        lig_rep = {}
        for k, v in comp.items():
            if type(v) is not torch.Tensor:
                prot_rep[k] = v
                lig_rep[k] = v
            else:
                # prot_idx = torch.arange(len(idx))[~idx].to(v.device)
                # lig_idx = torch.arange(len(idx))[idx].to(v.device)
                # prot_rep[k] = torch.index_select(v, 0, prot_idx)
                # lig_rep[k] = torch.index_select(v, 0, lig_idx)
                prot_rep[k] = v[~idx]
                lig_rep[k] = v[idx]

        return prot_rep, lig_rep


class GroupedModel(Model):
    """
    Subclass of the above `Model` for use with grouped data, eg multiple docked
    poses of the same molecule with the same protein. In addition to the
    `Representation` and `Strategy` modules in the `Model` class, `GroupedModel`
    also has a `Comination` module, that dictates how the `Model` predictions
    for each item in the group of data are combined.
    """

    def __init__(
        self,
        representation,
        strategy,
        combination,
        pred_readout=None,
        comb_readout=None,
    ):
        """
        The `representation`, `strategy`, and `pred_readout` options will be used
        to initialize the underlying `Model` object, while the `combination` and
        `comb_readout` modules will be applied to the output of the `Model` preds.

        Parameters
        ----------
        representation : Representation
            Representation object to get the representation of the input data.
        strategy : Strategy
            Strategy object to get convert the representations into energy preds.
        combination : Combination
            Combination object for combining the energy predictions.
        pred_readout : Readout
            Readout object for the energy predictions.
        comb_readout : Readout
            Readout object for the combination output.
        """
        super(GroupedModel, self).__init__(
            representation, strategy, pred_readout
        )
        self.combination = combination
        self.readout = comb_readout

    def forward(self, input_list):
        """
        Forward method for `GroupedModel` class. Will call the `forward` method
        of `Model` for each entry in `input_list`.

        Parameters
        ----------
        input_list : List[Tuple[Dict]]
            List of tuples of (complex representation, part representations)

        Returns
        -------
        torch.Tensor
            Combination of all predictions
        """
        ## Get predictions for all inputs in the list, and combine them in a
        ##  tensor (while keeping track of gradients)
        all_reps = torch.stack(
            [super(GroupedModel, self).forward(inp) for inp in input_list]
        )

        ## Combine each prediction according to `self.combination`
        comb_pred = self.combination(all_reps)
        if self.readout:
            return self.readout(comb_pred)
        else:
            return comb_pred


class Representation(torch.nn.Module):
    pass


class Strategy(torch.nn.Module):
    pass


class Combination(torch.nn.Module):
    pass


class Readout(torch.nn.Module):
    pass


class DeltaStrategy(Strategy):
    """
    Simple strategy for subtracting the sum of the individual component energies
    from the complex energy.
    """

    def __init__(self, energy_func, pic50=True):
        super(DeltaStrategy, self).__init__()
        self.energy_func: torch.nn.Module = energy_func
        self.pic50 = pic50

    def forward(self, comp, *parts):
        ## Calculat delta G
        return self.energy_func(comp) - sum(
            [self.energy_func(p) for p in parts]
        )


class ConcatStrategy(Strategy):
    """
    Strategy for combining the complex representation and parts representations
    in some learned manner, using sum-pooling to ensure permutation-invariance
    of the parts.
    """

    def __init__(self):
        super(ConcatStrategy, self).__init__()
        self.reduce_nn: torch.nn.Module = None

    def forward(self, comp, *parts):
        parts_size = sum([p.shape[1] for p in parts])
        if self.reduce_nn is None:
            ## These should already by representations, so initialize a Linear
            ##  module with appropriate input size
            input_size = comp.shape[1] + parts_size
            self.reduce_nn = torch.nn.Linear(input_size, 1)

        ## Move self.reduce_nn to appropriate torch device
        self.reduce_nn = self.reduce_nn.to(comp.device)

        ## Enumerate all possible permutations of parts and add together
        parts_cat = torch.zeros((1, parts_size), device=comp.device)
        for idxs in permutations(range(len(parts)), len(parts)):
            parts_cat += torch.cat([parts[i] for i in idxs], dim=1)

        ## Concat comp w/ permut-invariant parts representation
        full_embedded = torch.cat([comp, parts_cat], dim=1)

        return self.reduce_nn(full_embedded)


class BoltzmannCombination(Combination):
    """
    Combine a list of deltaG predictions according to their Boltzmann weight.
    """

    def __init__(self):
        super(BoltzmannCombination, self).__init__()

    def forward(self, predictions: torch.Tensor):
        return -kT * torch.log(torch.sum(torch.exp(-predictions)))


class PIC50Readout(Readout):
    """
    Readout implementation to convert delta G values to pIC50 values.
    """

    def __init__(self, T=298.0):
        """
        Initialize conversion with specified T (assume 298 K).

        Parameters
        ----------
        T : float, default=298
            Temperature for conversion.
        """
        super(PIC50Readout, self).__init__()

        from simtk.unit import BOLTZMANN_CONSTANT_kB as kB

        self.kT = (kB * T)._value

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
        ## IC50 value = exp(dG/kT) => pic50 = -log10(exp(dg/kT))
        return -torch.log10(torch.exp(delta_g / self.kT))
