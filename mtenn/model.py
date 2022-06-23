import torch

class Model():
    """
    Model object containing an ML model and a function for calling said model.

    `model_call` should have a signature of model_call(model, data), meaning
    that one data object will be passed to it at a time, along with the model.
    At a very minimum the `model_call` function will simply call the model on
    the data object, but if any operations are required on the data the
    `model_call` function should take care of them.
    """
    def __init__(self, model, model_call):
        super(Model, self).__init__()
        self.model: torch.nn.Module = model
        self.model_call: function = model_call

    def __call__(self, data):
        return self.model_call(self.model, data)
