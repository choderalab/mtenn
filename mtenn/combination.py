"""
Implementations for the ``Combination`` block in a :py:class:`GroupedModel
<mtenn.model.GroupedModel>`.

The ``Combination`` block is responsible for combining multiple single-pose model
predictions into a single multi-pose prediction. For more details on the implementation
of these classes, see the :ref:`comb-docs-page` docs page and the guide on
:ref:`new-combination-guide`.

All equations referenced here correspond to those in :ref:`implemented-combs`.
"""

import abc
import torch


class Combination(torch.nn.Module, abc.ABC):
    """
    Abstract base class for the ``Combination`` block. Any subclass needs to implement
    the ``forward`` method in order to be used.

    This class is designed to just be a wrapper around a ``torch.autograd.Function``
    subclass, as described in :ref:`the guide <new-combination-guide>`.
    """

    @abc.abstractmethod
    def forward(self, pred_list, grad_dict, param_names, *model_params):
        """
        This function signature should be the same for any ``Combination`` subclass
        implementation. The return values should be:

        * ``torch.Tensor``: Scalar-value tensor giving the final combined prediction

        * ``torch.Tensor``: Tensor of shape ``(n_predictions,)`` giving the input
          per-pose predictions. This is necessary for ``Pytorch`` to track the
          gradients of these predictions in the case of eg a cross-entropy loss on the
          per-pose predictions

        Parameters
        ----------
        pred_list: List[torch.Tensor]
            List of :math:`\mathrm{\Delta G}` predictions to be combined, shape of
            ``(n_predictions,)``
        grad_dict: dict[str, List[torch.Tensor]]
            Dict mapping from parameter name to list of gradients. Should contain
            ``n_model_parameters`` entries, with each entry mapping to a list of
            ``n_predictions`` tensors. Each of these tensors is a ``detach`` ed gradient
            so the shape of each tensor will depend on the model parameter it
            corresponds to, but the shapes of each tensor in any given entry should be
            identical
        param_names: List[str]
            List of parameter names. Should contain ``n_model_parameters`` entries,
            corresponding 1:1 with the keys in ``grad_dict``
        model_params: List[torch.Tensor]
            Actual parameters that we'll return the gradients for. Each param
            should be passed directly for the backward pass to
            work right. These tensors should correspond 1:1 with and should be in the
            same order as the entries in ``param_names`` (ie the ``i`` th entry in
            ``param_names`` should be the name of the ``i`` th model parameter in
            ``model_params``)
        """
        raise NotImplementedError("Must implement the `forward` method.")

    @staticmethod
    def split_grad_dict(grad_dict):
        """
        Helper method used by all ``Combination`` classes to split up the passed
        ``grad_dict`` for saving by context manager.

        Parameters
        ----------
        grad_dict : Dict[str, List[torch.Tensor]]
            Dict mapping from parameter name to list of gradients

        Returns
        -------
        List[str]
            Key in ``grad_dict`` corresponding 1:1 with the gradients
        List[torch.Tensor]
            Gradients from ``grad_dict`` corresponding 1:1 with the keys
        """
        # Deconstruct grad_dict to be saved for backwards
        grad_dict_keys = [
            k for k, grad_list in grad_dict.items() for _ in range(len(grad_list))
        ]
        grad_dict_tensors = [
            grad for grad_list in grad_dict.values() for grad in grad_list
        ]

        return grad_dict_keys, grad_dict_tensors

    @staticmethod
    def join_grad_dict(grad_dict_keys, grad_dict_tensors):
        """
        Helper method used by all ``Combination`` classes to reconstruct the
        ``grad_dict`` from keys and grad tensors.

        Parameters
        ----------
        grad_dict_keys : List[str]
            Key in ``grad_dict`` corresponding 1:1 with the gradients
        grad_dict_tensors : List[torch.Tensor]
            Gradients from ``grad_dict`` corresponding 1:1 with the keys

        Returns
        -------
        Dict[str, List[torch.Tensor]]
            Dict mapping from parameter name to list of gradients
        """
        # Reconstruct grad_dict
        grad_dict = {}
        for k, grad in zip(grad_dict_keys, grad_dict_tensors):
            try:
                grad_dict[k].append(grad)
            except KeyError:
                grad_dict[k] = [grad]

        return grad_dict


class MeanCombination(Combination):
    """
    Combine a list of predictions by taking the mean. See the docs for
    :py:class:`MeanCombinationFunc <mtenn.combination.MeanCombinationFunc>` for more
    details.
    """

    def forward(self, pred_list, grad_dict, param_names, *model_params):
        return MeanCombinationFunc.apply(
            pred_list, grad_dict, param_names, *model_params
        )


class MeanCombinationFunc(torch.autograd.Function):
    """
    Custom autograd function that will handle the gradient math for us for combining
    :math:`\mathrm{\Delta G}` predictions to their mean.

    .. math::

        \Delta \\text{G}(\\theta) = \\frac{1}{N}
        \\sum_{i=1}^{N} \\Delta \\text{G}_i (\\theta)

    See :ref:`mean-comb-imp` for more details on the math.

    """

    @staticmethod
    def forward(pred_list, grad_dict, param_names, *model_params):
        """
        Take the mean of all input :math:`\mathrm{\Delta G}` predictions.

        Parameters
        ----------
        pred_list: List[torch.Tensor]
            List of :math:`\mathrm{\Delta G}` predictions to be combined, shape of
            ``(n_predictions,)``
        grad_dict: dict[str, List[torch.Tensor]]
            Dict mapping from parameter name to list of gradients. Should contain
            ``n_model_parameters`` entries, with each entry mapping to a list of
            ``n_predictions`` tensors. Each of these tensors is a ``detach`` ed gradient
            so the shape of each tensor will depend on the model parameter it
            corresponds to, but the shapes of each tensor in any given entry should be
            identical
        param_names: List[str]
            List of parameter names. Should contain ``n_model_parameters`` entries,
            corresponding 1:1 with the keys in ``grad_dict``
        model_params: List[torch.Tensor]
            Actual parameters that we'll return the gradients for. Each param
            should be passed directly for the backward pass to
            work right. These tensors should correspond 1:1 with and should be in the
            same order as the entries in ``param_names`` (ie the ``i`` th entry in
            ``param_names`` should be the name of the ``i`` th model parameter in
            ``model_params``)

        Returns
        -------
        torch.Tensor
            Scalar-value tensor giving the mean of the input :math:`\mathrm{\Delta G}`
            predictions
        torch.Tensor
            Tensor of shape ``(n_predictions,)`` giving the input per-pose predictions
        """
        # Return mean of all preds
        all_preds = torch.stack(pred_list).flatten()
        final_pred = all_preds.mean(axis=None).detach()

        return final_pred, all_preds

    @staticmethod
    def setup_context(ctx, inputs, output):
        """
        Store data for backward pass.

        Parameters
        ----------
        ctx
            Pytorch context manager
        inputs : List
            List containing all the parameters that will get passed to ``forward``
        output : torch.Tensor
            Value returned from ``forward``
        """

        pred_list, grad_dict, param_names, *model_params = inputs

        grad_dict_keys, grad_dict_tensors = Combination.split_grad_dict(grad_dict)

        # Save non-Tensors for backward
        ctx.grad_dict_keys = grad_dict_keys
        ctx.param_names = param_names

        # Save Tensors for backward
        # Saving:
        #  * Predictions (1 tensor)
        #  * Grad tensors (N params * M poses tensors)
        #  * Model param tensors (N params tensors)
        ctx.save_for_backward(
            torch.stack(pred_list).flatten(),
            *grad_dict_tensors,
            *model_params,
        )

    @staticmethod
    def backward(ctx, comb_grad, pose_grads):
        """
        Compute and return gradients for each parameter.

        Parameters
        ----------
        ctx
            Pytorch context manager
        comb_grad : torch.Tensor
            Scalar-value tensor giving the
            :math:`\\frac{\\partial L}{\\partial \\Delta \\text{G}}` term from
            :eq:`comb-grad`
        pose_grads : torch.Tensor
            Tensor of shape ``(n_predictions,)``, giving the
            :math:`\\frac{\\partial L}{\\partial \\Delta \\text{G}_i}` terms from
            :eq:`pose-grad`
        """
        # Unpack saved tensors
        preds, *other_tensors = ctx.saved_tensors

        # First section of these tensors are the flattened lists of gradients from each
        #  individual pose or each model parameter
        grad_dict_tensors = other_tensors[: len(ctx.grad_dict_keys)]

        # Reconstruct dict mapping from model parameter name to list of gradient tensors
        # The ith entry in each list gives the gradient of the ith pose prediction wrt
        #  that model parameter
        grad_dict = Combination.join_grad_dict(ctx.grad_dict_keys, grad_dict_tensors)

        # Calculate final gradients for each parameter
        final_grads = {}
        for n, grad_list in grad_dict.items():
            # Compute the gradient contributions from any combined prediction loss,
            #  according to eqns (1), (4)
            cur_final_grad = comb_grad * torch.stack(grad_list, axis=-1).mean(axis=-1)

            # Make sure lengths match up (should always be true but just in case)
            if len(pose_grads) != len(grad_list):
                raise RuntimeError("Mismatch in gradient lengths.")

            # Compute the gradient contributions from any per-pose prediction loss,
            #  according to eqn (2)
            for pose_grad, param_grad in zip(pose_grads, grad_list):
                cur_final_grad += pose_grad * param_grad

            # Store total gradient for each parameter
            final_grads[n] = cur_final_grad.clone()

        # Return gradients for each of the model parameters that were passed in. Also
        #  need to return values for the other values that were passed to forward
        #  (pred_list, grad_dict, param_names), but these don't get gradients so we just
        #  return None
        return_vals = [None] * 3 + [final_grads[n] for n in ctx.param_names]
        return tuple(return_vals)


class MaxCombination(Combination):
    """
    Approximate max/min of the predictions using the LogSumExp function for smoothness.
    See the docs for :py:class:`MaxCombinationFunc
    <mtenn.combination.MaxCombinationFunc>` for more details.
    """

    def __init__(self, negate_preds=True, pred_scale=1000.0):
        """
        Parameters
        ----------
        negate_preds : bool, default=True
            Negate the predictions before calculating the LSE, effectively finding
            the min. Preds are negated again before being returned
        pred_scale : float, default=1000.0
            Fixed positive value to scale predictions by before taking the LSE. This
            tightens the bounds of the LSE approximation
        """
        super(MaxCombination, self).__init__()

        self.negate_preds = negate_preds
        self.pred_scale = pred_scale

    def __repr__(self):
        return f"MaxCombination(negate_preds={self.negate_preds}, pred_scale={self.pred_scale})"

    def __str__(self):
        return repr(self)

    def forward(self, pred_list, grad_dict, param_names, *model_params):
        return MaxCombinationFunc.apply(
            self.negate_preds,
            self.pred_scale,
            pred_list,
            grad_dict,
            param_names,
            *model_params,
        )


class MaxCombinationFunc(torch.autograd.Function):
    """
    Custom autograd function that will handle the gradient math for us for taking the
    max/min of the :math:`\mathrm{\Delta G}` predictions.

    For the ``forward`` pass, the final :math:`\mathrm{\Delta G}` prediction is
    calculated according to the following:

    .. math::

        n = \\begin{cases}
        -1 & \\text{negate_preds} \\\\
        \\phantom{-}1  & \\text{not negate_preds}
        \\end{cases}

    .. math::

        t &= \\text{pred_scale}

        \Delta G &= n \\frac{1}{t} \mathrm{ln} \sum_{n=1}^N \mathrm{exp} (n t \Delta G_n)

    The logic and math behind this scaling approach are detailed `here
    <https://en.wikipedia.org/wiki/LogSumExp#Properties>`_.

    See :ref:`max-comb-imp` for more details on the math.
    """

    @staticmethod
    def forward(
        negate_preds, pred_scale, pred_list, grad_dict, param_names, *model_params
    ):
        """
        Find the max/min of all input :math:`\mathrm{\Delta G}` predictions.

        Parameters
        ----------
        negate_preds: bool
            Negate the predictions before calculating the LSE, effectively finding
            the min. Preds are negated again before being returned
        pred_scale: float
            Fixed positive value to scale predictions by before taking the LSE. This
            tightens the bounds of the LSE approximation
        pred_list: List[torch.Tensor]
            List of :math:`\mathrm{\Delta G}` predictions to be combined, shape of
            ``(n_predictions,)``
        grad_dict: dict[str, List[torch.Tensor]]
            Dict mapping from parameter name to list of gradients. Should contain
            ``n_model_parameters`` entries, with each entry mapping to a list of
            ``n_predictions`` tensors. Each of these tensors is a ``detach`` ed gradient
            so the shape of each tensor will depend on the model parameter it
            corresponds to, but the shapes of each tensor in any given entry should be
            identical
        param_names: List[str]
            List of parameter names. Should contain ``n_model_parameters`` entries,
            corresponding 1:1 with the keys in ``grad_dict``
        model_params: List[torch.Tensor]
            Actual parameters that we'll return the gradients for. Each param
            should be passed directly for the backward pass to
            work right. These tensors should correspond 1:1 with and should be in the
            same order as the entries in ``param_names`` (ie the ``i`` th entry in
            ``param_names`` should be the name of the ``i`` th model parameter in
            ``model_params``)

        Returns
        -------
        torch.Tensor
            Scalar-value tensor giving the max/min of the input
            :math:`\mathrm{\Delta G}` predictions
        torch.Tensor
            Tensor of shape ``(n_predictions,)`` giving the input per-pose predictions
        """
        # The value of negate_preds tells us if we are finding the max or min. If True,
        #  we are finding the min and need to flip the sign of each individual
        #  prediction, as well as the final combined prediction (this is the value n
        #  described in the class docstring and associated implementation math section)
        negative_multiplier = -1 if negate_preds else 1

        # Combine all torch tensors so we don't need to keep doing it at each step
        all_preds = torch.stack(pred_list).flatten()

        # We use adj_preds here to store the adjusted per-pose prediction values. These
        #  values have been negated (if we are finding the min), and multiplied by our
        #  scale value, if given
        # These values correspond to the values inside the exponential in eqn (5) (and
        #  subsequent equations)
        adj_preds = negative_multiplier * pred_scale * all_preds.detach()

        # Although defining this intermediate value isn't as helpful/necessary in the
        #  forward pass, we do so anyway for consistency with the backward pass, where
        #  it will be necessary for numerical stability
        # This corresponds to eqn (6)
        Q = torch.logsumexp(adj_preds, dim=0)

        # Perform the inverse adjustments we applied to the per-pose predictions, giving
        #  us (approximately) the original value of the max/min per-pose prediction
        final_pred = (negative_multiplier * Q / pred_scale).detach()

        return final_pred, all_preds

    @staticmethod
    def setup_context(ctx, inputs, output):
        """
        Store data for backward pass.

        Parameters
        ----------
        ctx
            Pytorch context manager
        inputs : List
            List containing all the parameters that will get passed to ``forward``
        output : torch.Tensor
            Values returned from ``forward``
        """
        # Unpack the inputs
        (
            negate_preds,
            pred_scale,
            pred_list,
            grad_dict,
            param_names,
            *model_params,
        ) = inputs

        # Break the grad dict up into lists of keys and corresponding lists of gradients
        grad_dict_keys, grad_dict_tensors = Combination.split_grad_dict(grad_dict)

        # Save non-Tensors for backward
        ctx.negate_preds = negate_preds
        ctx.pred_scale = pred_scale
        ctx.grad_dict_keys = grad_dict_keys
        ctx.param_names = param_names

        # Save Tensors for backward
        # Saving:
        #  * Predictions (1 tensor of shape (n_predictions,))
        #  * Grad tensors (N params * M poses tensors, where all gradients corresponding
        #    to a given model parameter are adjacent, ie first M tensors are the
        #    per-pose gradients for the first model parameter, etc)
        #  * Model param tensors (N params tensors)
        ctx.save_for_backward(
            torch.stack(pred_list).flatten(),
            *grad_dict_tensors,
            *model_params,
        )

    @staticmethod
    def backward(ctx, comb_grad, pose_grads):
        """
        Compute and return gradients for each parameter.

        Parameters
        ----------
        ctx
            Pytorch context manager
        comb_grad : torch.Tensor
            Scalar-value tensor giving the
            :math:`\\frac{\\partial L}{\\partial \\Delta \\text{G}}` term from
            :eq:`comb-grad`
        pose_grads : torch.Tensor
            Tensor of shape ``(n_predictions,)``, giving the
            :math:`\\frac{\\partial L}{\\partial \\Delta \\text{G}_i}` terms from
            :eq:`pose-grad`
        """
        # Unpack saved tensors
        preds, *other_tensors = ctx.saved_tensors

        # First section of these tensors are the flattened lists of gradients from each
        #  individual pose or each model parameter
        grad_dict_tensors = other_tensors[: len(ctx.grad_dict_keys)]

        # Reconstruct dict mapping from model parameter name to list of gradient tensors
        # The ith entry in each list gives the gradient of the ith pose prediction wrt
        #  that model parameter
        grad_dict = Combination.join_grad_dict(ctx.grad_dict_keys, grad_dict_tensors)

        # Set negation multiplier for finding max/min (see docstring and associated
        #  implementation math section for more details)
        negative_multiplier = -1 if ctx.negate_preds else 1

        # We use adj_preds here to store the adjusted per-pose prediction values. These
        #  values have been negated (if we are finding the min), and multiplied by our
        #  scale value, if given
        # These values correspond to the values inside the exponential in eqn (5) (and
        #  subsequent equations)
        adj_preds = negative_multiplier * ctx.pred_scale * preds.detach()

        # Calculate our normalizing constant (eqn (6))
        Q = torch.logsumexp(adj_preds, dim=0)

        # Calculate final gradients for each parameter
        final_grads = {}
        for n, grad_list in grad_dict.items():
            # Compute the gradient contributions from any combined prediction loss,
            #  according to eqns (1), (9)
            cur_final_grad = comb_grad * (
                torch.stack(
                    [
                        grad * (pred - Q).exp()
                        for grad, pred in zip(grad_list, adj_preds)
                    ],
                    axis=-1,
                )
                .detach()
                .sum(axis=-1)
            )

            # Make sure lengths match up (should always be true but just in case)
            if len(pose_grads) != len(grad_list):
                raise RuntimeError("Mismatch in gradient lengths.")

            # Compute the gradient contributions from any per-pose prediction loss,
            #  according to eqn (2)
            for pose_grad, param_grad in zip(pose_grads, grad_list):
                cur_final_grad += pose_grad * param_grad

            # Store total gradient for each parameter
            final_grads[n] = cur_final_grad.clone()

        # Return gradients for each of the model parameters that were passed in. Also
        #  need to return values for the other values that were passed to forward
        #  (negate_preds, pred_scale, pred_list, grad_dict, param_names), but these
        #  don't get gradients so we just return None
        return_vals = [None] * 5 + [final_grads[n] for n in ctx.param_names]
        return tuple(return_vals)
