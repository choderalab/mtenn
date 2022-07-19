"""
Representation and strategy for e3nn model.
"""
from copy import deepcopy
import torch
from e3nn import o3
from e3nn.nn.models.gate_points_2101 import Network

from ..model import ConcatStrategy, DeltaStrategy, Model

class E3NN(Network):
    def __init__(self, model_kwargs, model=None):
        ## If no model is passed, construct E3NN model with model_kwargs, otherwise copy
        ##  all parameters and weights over
        if model is None:
            super(E3NN, self).__init__(**model_kwargs)
            self.parameters = model_kwargs
        else:
            # this will need changing to include  model features of e3nn
            atomref = model.atomref.weight.detach().clone()
            model_params = (model.hidden_channels, model.num_filters,
                model.num_interactions, model.num_gaussians,
                model.cutoff, model.max_num_neighbors,model.readout,
                model.dipole, model.mean, model.std, atomref)
            super(E3NN, self).__init__(*model_params)
            self.parameters = model_params
            self.load_state_dict(model.state_dict())

    def forward(self, data):
        x = super(E3NN, self).forward(data)
        copy = deepcopy(data)
        copy['x'] = torch.clone(x)
        return copy


    def _get_representation(self):
        """
        Input model, remove last layer.
        Parameters
        ----------
        model: E3NN
            e3nn model
        Returns
        -------
        E3NN
            Copied e3nn model with the last layer removed
        """

        ## Copy model so initial model isn't affected
        model_copy = deepcopy(self)
        ## Remove last layer
        model_copy.layers = model_copy.layers[:-1]
        model_copy.reduce_output = False

        return(model_copy)
    

    def _get_energy_func(self):
        """
        Return last layer of the model.
        Parameters
        ----------
        model: e3nn
            e3nn model
        Returns
        -------
        e3nn.nn.models.gate_points_2101.Network
            Copy of `model`'s last layer
        """

        final_conv  = deepcopy(self)
        
        conv_kwargs = deepcopy(self.parameters)
        conv_kwargs['layers'] = 0

        new_model = Network(**conv_kwargs)

        new_model.layers[0] = final_conv.layers[-1]

        return(new_model)

    def _get_delta_strategy(self):
        """
        Build a DeltaStrategy object based on the passed model.
        Parameters
        ----------
        model: E3NN
            e3nn model
        Returns
        -------
        DeltaStrategy
            DeltaStrategy built from `model`
        """

        return(DeltaStrategy(self._get_energy_func()))

    @staticmethod
    def get_model(model=None, model_kwargs=None, strategy: str='delta'):
        """
        Exposed function to build a Model object from a E3NN object. If none
        is provided, a default model is initialized.
        Parameters
        ----------
        model: E3NN, optional
            E3NN model to use to build the Model object. If left as none, a
            default model will be initialized and used
        model_kwargs: dict, optional
            Dictionary used to initialize E3NN model if model is not passed in
        strategy: str, default='delta'
            Strategy to use to combine representation of the different parts.
            Options are ['delta', 'concat']
        Returns
        -------
        Model
            Model object containing the desired Representation and Strategy
        """

        #
        if model is None:
            model = E3NN(model_kwargs)

        ## First get representation module
        representation = model._get_representation()

        ## Construct strategy module based on model and
        ##  representation (if necessary)
        if strategy == 'delta':
            strategy = model._get_delta_strategy()
        elif strategy == 'concat':
            strategy = ConcatStrategy()

        return(Model(representation, strategy))
