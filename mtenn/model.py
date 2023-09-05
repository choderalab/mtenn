from copy import deepcopy
from itertools import permutations
import os
import torch
from typing import Optional


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
            return self.readout(energy_val)
        else:
            return energy_val

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
            self.combination(super(GroupedModel, self).forward(inp), self)

        ## Combine each prediction according to `self.combination`
        comb_pred = self.combination.predict(self)
        if self.readout:
            return self.readout(comb_pred)
        else:
            return comb_pred


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


class Representation(torch.nn.Module):
    pass


class Strategy(torch.nn.Module):
    pass


class Combination(torch.nn.Module):
    def __init__(self):
        """
        Stuff that will be common among all Combination subclasses.
        """
        super(Combination, self).__init__()

        self.gradients = {}
        self.predictions = []

    def forward(self, prediction: torch.Tensor, model: torch.nn.Module):
        """
        Takes a prediction and model, and tracks the pred and the gradient of the pred
        wrt model parameters. This part should be the same for all Combination methdos,
        so we can put it in the base class.

        Parameters
        ----------
        prediction : torch.Tensor
            Model prediction
        model : torch.nn.Module
            The model being trained
        """
        # Track prediction
        self.predictions.append(prediction.detach())

        # Don't do anything with gradients if model is in eval mode
        if not model.training:
            return

        # Get gradients (zero first to get rid of any existing)
        model.zero_grad()
        prediction.backward()
        for n, p in model.named_parameters():
            try:
                self.gradients[n].append(p.grad.detach())
            except KeyError:
                self.gradients[n] = [p.grad.detach()]

    def predict(self):
        raise NotImplementedError(
            "A Combination class must have the `predict` method implemented."
        )


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


class MeanCombination(Combination):
    """
    Combine a list of predictions by taking the mean.
    """

    def __init__(self):
        super(MeanCombination, self).__init__()

    def predict(self, model: torch.nn.Module):
        """
        Returns the mean of all stored predictions, and appropriately sets the model
        parameter grads for an optimizer step.

        Parameters
        ----------
        model : torch.nn.Module
            The model being trained

        Returns
        -------
        torch.Tensor
            Combined prediction (mean of all stored preds)
        """
        # Return mean of all preds
        final_pred = torch.stack(self.predictions).mean(axis=None).detach()

        if model.training:
            # Calculate final gradient (derivation details are in README_COMBINATION)
            for n, p in model.named_parameters():
                p.grad = torch.stack(self.gradients[n], axis=-1).mean(axis=-1)

            # Reset internal trackers
            self.gradients = {}
        self.predictions = []

        return final_pred


class MaxCombination(Combination):
    """
    Approximate max/min of the predictions using the LogSumExp function for smoothness.
    """

    def __init__(self, neg=True, scale=1000.0):
        """
        Parameters
        ----------
        neg : bool, default=True
            Negate the predictions before calculating the LSE, effectively finding
            the min. Preds are negated again before being returned
        scale : float, default=1000.0
            Fixed positive value to scale predictions by before taking the LSE. This
            tightens the bounds of the LSE approximation
        """
        super(MaxCombination, self).__init__()

        self.neg = -1 * neg
        self.scale = scale

    def predict(self, model: torch.nn.Module):
        """
        Returns the max/min of all stored predictions, and appropriately sets the model
        parameter grads for an optimizer step.

        Parameters
        ----------
        model : torch.nn.Module
            The model being trained

        Returns
        -------
        torch.Tensor
            Combined prediction (LSE max approximation of all stored preds)
        """
        # Calculate once for reuse later
        adj_preds = self.neg * self.scale * torch.stack(self.predictions).flatten()
        Q = torch.logsumexp(adj_preds, dim=0)
        # Calculate the actual prediction
        final_pred = (self.neg * Q / self.scale).detach()

        if model.training:
            # Calculate final gradients for each parameter
            final_grads = {}
            for n, p in self.gradients.items():
                final_grads[n] = (
                    torch.stack(
                        [g * (pred - Q).exp() for g, pred in zip(p, adj_preds)],
                        axis=-1,
                    )
                    .detach()
                    .sum(axis=-1)
                )

            # Set weights gradients
            for n, p in model.named_parameters():
                try:
                    p.grad = final_grads[n]
                except RuntimeError as e:
                    print(n, p.grad.shape, final_grads[n].shape, flush=True)
                    raise e

            # Reset internal trackers
            self.gradients = {}
        self.predictions = []

        return final_pred


class BoltzmannCombination(Combination):
    """
    Combine a list of deltaG predictions according to their Boltzmann weight.
    Treat energy in implicit kT units.
    """

    def __init__(self):
        super(BoltzmannCombination, self).__init__()

    def predict(self, model: torch.nn.Module):
        """
        Returns the Boltzmann weighted average of all stored predictions, and
        appropriately sets the model parameter grads for an optimizer step.

        Parameters
        ----------
        model : torch.nn.Module
            The model being trained

        Returns
        -------
        torch.Tensor
            Combined prediction (Boltzmann-weighted average)
        """
        # Save for later so we don't have to keep redoing this
        adj_preds = -torch.stack(self.predictions).flatten().detach()

        # First calculate the normalization factor
        Q = torch.logsumexp(adj_preds, dim=0)

        # Calculate w
        w = (adj_preds - Q).exp()

        # Calculate final pred
        final_pred = torch.dot(w, -adj_preds)

        if model.training:
            # Calculate dQ/d_theta
            dQ = {
                n: -torch.stack(
                    [(p - Q).exp() * g for p, g in zip(adj_preds, grads)], axis=-1
                ).sum(axis=-1)
                for n, grads in self.gradients.items()
            }

            # Calculate dw/d_theta
            dw = {
                n: [(p - Q).exp() * (-g - dQ[n]) for p, g in zip(adj_preds, grads)]
                for n, grads in self.gradients.items()
            }

            # Calculate final grads
            final_grads = {}
            for n, p in self.gradients.items():
                final_grads[n] = (
                    torch.stack(
                        [
                            w_grad * -pred + w_val * grad
                            for w_grad, pred, w_val, grad in zip(dw[n], adj_preds, w, p)
                        ],
                        axis=-1,
                    )
                    .detach()
                    .sum(axis=-1)
                )

            # Set weights gradients
            for n, p in model.named_parameters():
                try:
                    p.grad = final_grads[n]
                except RuntimeError as e:
                    print(n, p.grad.shape, final_grads[n].shape, flush=True)
                    raise e

            # Reset internal trackers
            self.gradients = {}
        self.predictions = []

        return final_pred


class PIC50Readout(Readout):
    """
    Readout implementation to convert delta G values to pIC50 values. This new
    implementation assumes implicit energy units, WHICH WILL INVALIDATE MODELS TRAINED
    PRIOR TO v0.3.0.
    Assuming implicit energy units:
        deltaG = ln(Ki)
        Ki = exp(deltaG)
    Using the Cheng-Prusoff equation:
        Ki = IC50 / (1 + [S]/Km)
        exp(deltaG) = IC50 / (1 + [S]/Km)
        IC50 = exp(deltaG) * (1 + [S]/Km)
        pIC50 = -log10(exp(deltaG) * (1 + [S]/Km))
        pIC50 = -log10(exp(deltaG)) - log10(1 + [S]/Km)
        pIC50 = -ln(exp(deltaG))/ln(10) - log10(1 + [S]/Km)
        pIC50 = -deltaG/ln(10) - log10(1 + [S]/Km)
    Estimating Ki as the IC50 value:
        Ki = IC50
        IC50 = exp(deltaG)
        pIC50 = -log10(exp(deltaG))
        pIC50 = -ln(exp(deltaG))/ln(10)
        pIC50 = -deltaG/ln(10)
    """

    def __init__(self, substrate: Optional[float] = None, Km: Optional[float] = None):
        """
        Initialize conversion with specified substrate concentration and Km. If either
        is left blank, the IC50 approximation will be used.

        Parameters
        ----------
        substrate : float, optional
            Substrate concentration for use in the Cheng-Prusoff equation. Assumed to be
            in the same units as Km
        Km : float, optional
            Km value for use in the Cheng-Prusoff equation. Assumed to be in the same
            units as substrate
        """
        super(PIC50Readout, self).__init__()

        self.substrate = substrate
        self.Km = Km

        if substrate and Km:
            self.cp_val = 1 + substrate / Km
        else:
            self.cp_val = None

    def __repr__(self):
        return f"PIC50Readout(substrate={self.substrate}, Km={self.Km})"

    def __str__(self):
        return repr(self)

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
        pic50 = -delta_g / torch.log(torch.tensor(10, dtype=delta_g.dtype))
        # Using Cheng-Prusoff
        if self.cp_val:
            pic50 -= torch.log10(torch.tensor(self.cp_val, dtype=delta_g.dtype))

        return pic50
