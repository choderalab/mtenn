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

    def __call__(self, data):
        return self.model_call(self.model, data)

    def get_representation(self, input):
        """
        Takes system topolgy and coordinates and returns Nxhidden dimension
        representation.

        Parameters
        ----------

        Returns
        -------
        """

    def forward(self, input0, input1):
        rep0 = self.get_representation(input0)
        rep1 = self.get_representation(input1)

        result = self.strategy(rep0, rep1)

class Representation(torch.nn.Module):
    pass

class Strategy(torch.nn.Module):
    pass

class DeltaStrategy(Strategy):
    def __init__(self, energy_func):
        self.energy_func: torch.nn.Module = energy_func

    def forward(self, complex, *parts):
        return(self.energy_func(complex)
            - sum([self.energy_func(p) for p in parts]))

class ConcatStrategy(Strategy):
    def __init__(self, in_nodes):
        self.reduce_nn: torch.nn.Module = torch.nn.Linear(in_nodes, 1)

    def forward(self, complex, *parts):
        if self.reduce_nn is None:
            self._init_nn([len(complex)] + [len(p) for p in parts])
        ## Enumerate all possible enumerations of parts + push through reduce_nn
        for
        ## Concat complex w/ permut-invariant parts representation
        full_embedded = torch.concat([complex, parts_embedded])
        return(self.reduce_nn(full_embedded))
