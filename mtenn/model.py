from itertools import permutations
import torch

class Model(torch.nn.Module):
    """
    Model object containing an ML model and a function for calling said model.

    `model_call` should have a signature of model_call(model, data), meaning
    that one data object will be passed to it at a time, along with the model.
    At a very minimum the `model_call` function will simply call the model on
    the data object, but if any operations are required on the data the
    `model_call` function should take care of them.
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
        complex_rep = self.get_representation(*comp)
        parts_rep = [self.get_representation(*p) for p in parts]

        return(self.strategy(complex_rep, *parts_rep))

class Representation(torch.nn.Module):
    pass

class Strategy(torch.nn.Module):
    pass

class DeltaStrategy(Strategy):
    def __init__(self, energy_func):
        super(DeltaStrategy, self).__init__()
        self.energy_func: torch.nn.Module = energy_func

    def forward(self, comp, *parts):
        return(self.energy_func(comp)
            - sum([self.energy_func(p) for p in parts]))

class ConcatStrategy(Strategy):
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
