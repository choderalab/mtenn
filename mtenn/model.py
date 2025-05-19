"""
This module contains the actual models that are used for making predictions. More
information on how everything here works is in the :ref:`docs page <model-docs-page>`.
"""

import os
import torch

from mtenn.combination import Combination
from mtenn.representation import Representation
from mtenn.strategy import ComplexOnlyStrategy, Strategy
from mtenn.readout import Readout


class Model(torch.nn.Module):
    """
    Model object containing a ``Representation`` block that will take an input
    and convert it into some representation, and a ``Strategy`` block that will
    take a complex representation and any number of constituent "part"
    representations, and convert to a final scalar value.
    """

    def __init__(self, representation, strategy, readout=None, fix_device=False):
        """
        Build a ``Model``.

        Parameters
        ----------
        representation : Representation
            ``Representation`` block for this model
        strategy : Strategy
            ``Strategy`` block for this model
        readout : Readout, optional
            ``Readout`` block for this model
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
        Pass a structure through the model's ``Representation`` block. All arguments are
        passed directly to ``self.representation``.
        """

        return self.representation(*args, **kwargs)

    def forward(self, comp, *parts):
        """
        Handles all the logic detailed in the :ref:`docs page <single-pose-model-docs>`.

        Parameters
        ----------
        comp : dict
            Complex structure that will be passed to the ``Representation`` block
        part : list[dict], optional
            Structures for all individual parts of the complex (eg ligand and protein
            separately). If this is not passed, the constituent parts will be
            automatically parsed from ``comp``

        Returns
        -------
        torch.Tensor
            Final model prediction. If the model has a ``readout``, this value will have
            the ``readout`` applied
        list[torch.Tensor]
            A list containing only the pre-``readout`` model prediction. This value is
            returned mainly to align the signatures for the single- and multi-pose
            models
        """
        # This implementation of the forward function assumes the
        #  get_representation function takes a single data object
        tmp_comp = self._fix_device(comp)
        complex_rep = self.get_representation(tmp_comp)

        if isinstance(self.strategy, ComplexOnlyStrategy):
            parts_rep = []
        else:
            if len(parts) == 0:
                parts = Model._split_parts(tmp_comp)
            parts_rep = [self.get_representation(self._fix_device(p)) for p in parts]

        energy_val = self.strategy(complex_rep, *parts_rep)
        if self.readout:
            return self.readout(energy_val), [energy_val]
        else:
            return energy_val, [energy_val]

    def _fix_device(self, data):
        """
        Make sure that the pose tensors are on the same device as the model before
        attempting to call the model. Note that if ``self.fix_device`` is ``False``,
        this function does nothing. Also note that this function uses the torch ``to``
        function, which means that if a tensor is on the wrong device, a copy of the
        tensor will be returned, whereas if the tensor is already on the correct device
        the original tensor will be returned.

        Parameters
        ----------
        data : dict
            Structure pose

        Returns
        -------
        dict
            New dict with all tensors on the appropriate device
        """
        # We'll call this on everything for uniformity, but if we fix_device is
        #  False we can just return
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
        comp : dict[str, object]
            Dictionary representing the complex data object. Must have "lig" as
            a key that contains the index for splitting the data.

        Returns
        -------
        dict[str, object]
            Protein representation
        dict[str, object]
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
    Subclass of :py:class:`Model <mtenn.model.Model>` for use with multi-pose data, eg
    multiple docked poses of the same molecule with the same protein. In addition to the
    ``Representation`` and ``Strategy`` blocks in the ``Model`` class, ``GroupedModel``
    also has a ``Combination`` block that dictates how the individual ``Model``
    predictions for each item in the group of data are combined.
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
        The ``representation``, ``strategy``, and ``pred_readout`` args will be used
        to initialize the underlying :py:class:`Model <mtenn.model.Model>`, while the
        ``combination`` and ``comb_readout`` args will be applied to the output of the
        individual pose predictions.

        Parameters
        ----------
        representation : Representation
            ``Representation`` block for this model
        strategy : Strategy
            ``Strategy`` block for this model
        combination : Combination
            ``Combination`` block for this model
        pred_readout : Readout, optional
            ``Readout`` block for the individual pose predictions
        comb_readout : Readout, optional
            ``Readout`` block for the combined pose
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
        Handles all the logic detailed in the :ref:`docs page <multi-pose-model-docs>`.

        Parameters
        ----------
        input_list : list[tuple[dict]]
            List of tuples of (complex representation, part representations)

        Returns
        -------
        torch.Tensor
            Final multi-pose model prediction
        list[torch.Tensor]
            A list containing the pre-``pred_readout`` prediction for each entry in
            ``input_list``
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
        comb_pred, comb_pred_list = self.combination(
            pred_list, grad_dict, param_names, *model_params
        )

        if self.comb_readout:
            return self.comb_readout(comb_pred), comb_pred_list
        else:
            return comb_pred, comb_pred_list


class LigandOnlyModel(Model):
    """
    A ligand-only version of the ``Model``. In this case, the ``representation`` block
    will hold the entire model, while the ``strategy`` block will simply be set as an
    Identity module.
    """

    def __init__(self, model, readout=None, fix_device=False):
        """
        Build a ``LigandOnlyModel``.

        Parameters
        ----------
        model
            This can be any kind of model that will go from a single input
            representation to a prediction (eg a
            :py:class:`GAT <mtenn.conversion_utils.gat.GAT` instance)
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
        """
        Handles all the logic detailed in the :ref:`docs page <ligand-only-model-docs>`.

        Parameters
        ----------
        rep
            Whatever input representation the unerlying model takes

        Returns
        -------
        torch.Tensor
            Final model prediction. If the model has a ``readout``, this value will have
            the ``readout`` applied
        list[torch.Tensor]
            A list containing only the pre-``readout`` model prediction. This value is
            returned mainly to align the signatures for the single- and multi-pose
            models
        """
        # This implementation of the forward function assumes the
        #  get_representation function takes a single data object
        tmp_rep = self._fix_device(rep)
        pred = self.get_representation(tmp_rep)

        if self.readout:
            return self.readout(pred), [pred]
        else:
            return pred, [pred]


class SplitModel(torch.nn.Module):
    """
    Model object containing a ``Representation`` block that will take an input
    and convert it into some representation, and a ``Strategy`` block that will
    take a complex representation and any number of constituent "part"
    representations, and convert to a final scalar value.
    """

    def __init__(
        self,
        complex_representation,
        strategy,
        ligand_representation=None,
        protein_representation=None,
        readout=None,
        fix_device=False,
    ):
        """
        Build a ``Model``.

        Parameters
        ----------
        complex_representation : Representation
            ``Representation`` block for the complex
        strategy : Strategy
            ``Strategy`` block for this model
        ligand_representation : Representation, optional
            ``Representation`` block for the ligand. Leave unset to use the
            ``Representation`` block in ``complex_representation``
        protein_representation : Representation, optional
            ``Representation`` block for the protein. Leave unset to use the
            ``Representation`` block in ``complex_representation``
        readout : Readout, optional
            ``Readout`` block for this model
        fix_device: bool, default=False
            If True, make sure the input is on the same device as the model,
            copying over as necessary.
        """
        super(SplitModel, self).__init__()
        self.complex_representation: Representation = complex_representation
        if ligand_representation is None:
            ligand_representation = complex_representation
        if protein_representation is None:
            protein_representation = complex_representation
        self.ligand_representation: Representation = ligand_representation
        self.protein_representation: Representation = protein_representation
        self.strategy: Strategy = strategy
        self.readout: Readout = readout

        self.fix_device = fix_device

    def get_representation(self, *args, **kwargs):
        """
        Pass a structure through the model's ``Representation`` block. All arguments are
        passed directly to ``self.representation``.
        """

        return self.representation(*args, **kwargs)

    def forward(self, comp, prot=None, lig=None):
        """
        Handles all the logic detailed in the :ref:`docs page <single-pose-model-docs>`.
        This class assumes the only data in ``parts`` is the protein and ligand, in that
        order.

        Parameters
        ----------
        comp : dict
            Complex structure that will be passed to the ``Representation`` block
        part : list[dict], optional
            Structures for protein and ligand. If this is not passed, the constituent parts will be
            automatically parsed from ``comp``

        Returns
        -------
        torch.Tensor
            Final model prediction. If the model has a ``readout``, this value will have
            the ``readout`` applied
        list[torch.Tensor]
            A list containing only the pre-``readout`` model prediction. This value is
            returned mainly to align the signatures for the single- and multi-pose
            models
        """
        # This implementation of the forward function assumes the
        #  get_representation function takes a single data object
        tmp_comp = self._fix_device(comp)
        complex_rep = self.complex_representation(tmp_comp)

        if isinstance(self.strategy, ComplexOnlyStrategy):
            lig_rep = None
            prot_rep = None
        else:
            if (prot is None) and (lig is None):
                prot, lig = Model._split_parts(tmp_comp)

            lig_rep = self.ligand_representation(self._fix_device(lig))
            prot_rep = self.protein_representation(self._fix_device(prot))

        energy_val = self.strategy(comp=complex_rep, prot=prot_rep, lig=lig_rep)
        if self.readout:
            return self.readout(energy_val), [energy_val]
        else:
            return energy_val, [energy_val]

    def _fix_device(self, data):
        """
        Make sure that the pose tensors are on the same device as the model before
        attempting to call the model. Note that if ``self.fix_device`` is ``False``,
        this function does nothing. Also note that this function uses the torch ``to``
        function, which means that if a tensor is on the wrong device, a copy of the
        tensor will be returned, whereas if the tensor is already on the correct device
        the original tensor will be returned.

        Parameters
        ----------
        data : dict
            Structure pose

        Returns
        -------
        dict
            New dict with all tensors on the appropriate device
        """
        # We'll call this on everything for uniformity, but if we fix_device is
        #  False we can just return
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
        comp : dict[str, object]
            Dictionary representing the complex data object. Must have "lig" as
            a key that contains the index for splitting the data.

        Returns
        -------
        dict[str, object]
            Protein representation
        dict[str, object]
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
