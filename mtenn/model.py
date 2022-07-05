from itertools import permutations
import torch

class Model(torch.nn.Module):
    """
    Model object containing a `representation` Module that will take an input
    and convert it into some representation, and a `strategy` module that will
    take a complex representation and any number of constituent "part"
    representations, and convert to a final scalar value.
    """
    def __init__(self, representation, strategy):
        super(Model, self).__init__()
        self.representation: Representation = representation
        self.strategy: Strategy = strategy

    def get_representation(self, *args, **kwargs):
        """
        Takes system topolgy and coordinates and returns Nxhidden dimension
        representation.

        Parameters
        ----------

        Returns
        -------
        """

        return(self.representation(*args, **kwargs))

    def forward(self, comp, *parts):
        ## This implementation of the forward function assumes the
        ##  get_representation function takes a single data object
        complex_rep = self.get_representation(comp)
        parts_rep = [self.get_representation(p) for p in parts]

        return(self.strategy(complex_rep, *parts_rep))

class Representation(torch.nn.Module):
    pass

class Strategy(torch.nn.Module):
    pass

class DeltaStrategy(Strategy):
    """
    Simple strategy for subtracting the sum of the individual component energies
    from the complex energy.
    """
    def __init__(self, energy_func):
        super(DeltaStrategy, self).__init__()
        self.energy_func: torch.nn.Module = energy_func

    def forward(self, comp, *parts):
        return(self.energy_func(comp)
            - sum([self.energy_func(p) for p in parts]))

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

        ## Enumerate all possible permutations of parts + add together
        parts_cat = torch.zeros((1,parts_size))
        for idxs in permutations(range(len(parts)), len(parts)):
            parts_cat += torch.cat([parts[i] for i in idxs], dim=1)

        ## Concat comp w/ permut-invariant parts representation
        full_embedded = torch.cat([comp, parts_cat], dim=1)

        return(self.reduce_nn(full_embedded))
