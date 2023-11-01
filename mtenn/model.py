import os
import torch

from mtenn.combination import Combination
from mtenn.representation import Representation
from mtenn.strategy import Strategy
from mtenn.readout import Readout


class Model(torch.nn.Module):
    """
    Model object containing a `representation` Module that will take an input
    and convert it into some representation, and a `strategy` module that will
    take a complex representation and any number of constituent "part"
    representations, and convert to a final scalar value.
    """

    def __init__(self, representation, strategy, readout=None, fix_device=False):
        """
        Parameters
        ----------
        fix_device: bool, default=False
            If True, make sure the input is on the same device as the model,
            copying over as necessary.
        """
        super(Model, self).__init__()
        self.representation: Representation = representation
        self.strategy: Strategy = strategy
        self.readout: Readout = readout

        self.fix_device = fix_device

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
        tmp_comp = self._fix_device(comp)
        complex_rep = self.get_representation(tmp_comp)

        if len(parts) == 0:
            parts = Model._split_parts(tmp_comp)
        parts_rep = [self.get_representation(self._fix_device(p)) for p in parts]

        energy_val = self.strategy(complex_rep, *parts_rep)
        if self.readout:
            return self.readout(energy_val), [energy_val]
        else:
            return energy_val, [energy_val]

    def _fix_device(self, data):
        ## We'll call this on everything for uniformity, but if we fix_deivec is
        ##  False we can just return
        if not self.fix_device:
            return data

        device = next(self.parameters()).device
        tmp_data = {}
        for k, v in data.items():
            try:
                tmp_data[k] = v.to(device)
            except AttributeError:
                tmp_data[k] = v

        return tmp_data

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
        fix_device=False,
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
        pred_readout : Readout, optional
            Readout object for the energy predictions.
        comb_readout : Readout, optional
            Readout object for the combination output.
        fix_device: bool, default=False
            If True, make sure the input is on the same device as the model,
            copying over as necessary.
        """
        super(GroupedModel, self).__init__(
            representation, strategy, pred_readout, fix_device
        )
        self.combination: Combination = combination
        self.comb_readout: Readout = comb_readout

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
        # Get predictions for all inputs in the list, and combine them in a
        #  tensor (while keeping track of gradients)
        pred_list = []
        grad_dict = {}
        for i, inp in enumerate(input_list):
            if "MTENN_VERBOSE" in os.environ:
                print(f"pose {i}", flush=True)
                print(
                    "size",
                    ", ".join(
                        [
                            f"{k}: {v.shape} ({v.dtype})"
                            for k, v in inp.items()
                            if type(v) is torch.Tensor
                        ]
                    ),
                    sum([len(p.flatten()) for p in self.parameters()]),
                    f"{torch.cuda.memory_allocated():,}",
                    flush=True,
                )
            # First get prediction
            pred, _ = super().forward(inp)
            pred_list.append(pred.detach())

            # Get gradient per sample
            self.zero_grad()
            pred.backward()
            for n, p in self.named_parameters():
                try:
                    grad_dict[n].append(p.grad.detach())
                except KeyError:
                    grad_dict[n] = [p.grad.detach()]
        # Zero grads again just to make sure nothing gets accumulated
        self.zero_grad()

        # Separate out param names and params
        param_names, model_params = zip(*self.named_parameters())
        comb_pred = self.combination(pred_list, grad_dict, param_names, *model_params)

        if self.comb_readout:
            return self.comb_readout(comb_pred), pred_list
        else:
            return comb_pred, pred_list


class LigandOnlyModel(Model):
    """
    A ligand-only version of the Model. In this case, the `representation` block will
    hold the entire model, while the `strategy` block will simply be set as an Identity
    module.
    """

    def __init__(self, model, readout=None, fix_device=False):
        """
        Parameters
        ----------
        fix_device: bool, default=False
            If True, make sure the input is on the same device as the model,
            copying over as necessary.
        """
        super(LigandOnlyModel, self).__init__(
            representation=model,
            strategy=torch.nn.Identity(),
            readout=readout,
            fix_device=fix_device,
        )

    def forward(self, rep):
        ## This implementation of the forward function assumes the
        ##  get_representation function takes a single data object
        tmp_rep = self._fix_device(rep)
        pred = self.get_representation(tmp_rep)

        if self.readout:
            return self.readout(pred)
        else:
            return pred
