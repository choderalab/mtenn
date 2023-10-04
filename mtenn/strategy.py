from itertools import permutations
import torch


class Strategy(torch.nn.Module):
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
        return self.energy_func(comp) - sum([self.energy_func(p) for p in parts])


class ConcatStrategy(Strategy):
    """
    Strategy for combining the complex representation and parts representations
    in some learned manner, using sum-pooling to ensure permutation-invariance
    of the parts.
    """

    def __init__(self, extract_key=None):
        """
        Parameters
        ----------
        extract_key : str, optional
            Key to use to extract representation from a dict
        """
        super(ConcatStrategy, self).__init__()
        self.reduce_nn: torch.nn.Module = None
        self.extract_key = extract_key

    def forward(self, comp, *parts):
        ## Extract representation from dict
        if self.extract_key:
            comp = comp[self.extract_key]
            parts = [p[self.extract_key] for p in parts]

        ## Flatten tensors
        comp = comp.flatten()
        parts = [p.flatten() for p in parts]

        parts_size = sum([len(p) for p in parts])
        if self.reduce_nn is None:
            ## These should already by representations, so initialize a Linear
            ##  module with appropriate input size
            input_size = len(comp) + parts_size
            self.reduce_nn = torch.nn.Linear(input_size, 1)

        ## Move self.reduce_nn to appropriate torch device
        self.reduce_nn = self.reduce_nn.to(comp.device)

        ## Enumerate all possible permutations of parts and add together
        parts_cat = torch.zeros((parts_size), device=comp.device)
        for idxs in permutations(range(len(parts)), len(parts)):
            parts_cat += torch.cat([parts[i] for i in idxs])

        ## Concat comp w/ permut-invariant parts representation
        full_embedded = torch.cat([comp, parts_cat])

        return self.reduce_nn(full_embedded)
