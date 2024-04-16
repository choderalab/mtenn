.. _new-combination-guide:

Implementing a new Combination
==============================

This guide will assume that you've already read the :doc:`../docs/combination` docs page, so head there first if you haven't.
In this guide, we will walk through both the math and the software engineering that was done to implement :py:class:`MaxCombination <mtenn.combination.MaxCombination>`.
These steps should hopefully be illustrative enough to enable anyone to implement their own ``Combination`` method.

.. _new-comb-math:

Math
----
We will use the LogSumExp function (LSE) as a differentiable smooth maximum:

.. math::

    \mathrm{LSE} (x_1, ..., x_n) = \mathrm{log} \sum \mathrm{exp}(x_i)

In the actual implementation in :py:class:`mtenn.combination.MaxCombination`, we include an optional scaling parameter and an option to find the min instead of the max by flipping all the signs, however in this example we will stick with the vanilla LSE function to simplify the math a bit.

As a reminder, the key function that we need to solve for analytically is the function (:math:`g`) that relates the individual single-pose predictions (:math:`\hat{y}_i(\theta)`) and pose prediction gradients (:math:`\frac{\partial \hat{y}_i}{\partial \theta}`) to the gradient of our overall multi-pose prediction (:math:`\frac{\partial \hat{y}}{\partial \theta}`):

.. math::

    \frac{\partial \hat{y}}{\partial \theta} = g( \hat{y}_1, ..., \hat{y}_n, \frac{\partial \hat{y}_1}{\partial \theta}, ..., \frac{\partial \hat{y}_n}{\partial \theta} )

In our LSE example, we have that

.. math::
    \hat{y}(\theta) = \mathrm{LSE} (\hat{y}_1(\theta), ..., \hat{y}_n(\theta)) = \mathrm{log} \sum \mathrm{exp}(\hat{y}_i(\theta))

and

.. math::

    \frac{\partial \hat{y}(\theta)}{\partial \theta} =
    \frac{1}{\sum \mathrm{exp}(\hat{y}_i(\theta))}
    \sum \left[ \frac{\partial \hat{y}_i(\theta)}{\partial \theta} \mathrm{exp}(\hat{y}_i(\theta)) \right]

At this point we're essentially done, as each of the :math:`\frac{\partial \hat{y}_i(\theta)}{\partial \theta}` terms will be calculated automatically inside our multi-pose model using ``torch.autograd``, and the :math:`\hat{y}_i` terms are just the predictions we've already generated.

Although this formulation is fine from a theoretical point of view, it poses some challenges in computing.
The exponentiation of the single-pose predictions can lead to the gradients exploding/vanishing, so we introduce a substitution that will allow us to take advantage of the numerically stable ``torch.logsumexp`` function:

.. math::

    Q = \mathrm{LSE}(\hat{y}_i(\theta))

Substituting this back into our previous equation, we get

.. math::


    \frac{\partial \hat{y}(\theta)}{\partial \theta} =
    \frac{1}{\mathrm{exp}(Q)}
    \sum \left[ \frac{\partial \hat{y}_i(\theta)}{\partial \theta} \mathrm{exp}(\hat{y}_i(\theta)) \right]

which can be further rearranged to

.. math::

    \frac{\partial \hat{y}(\theta)}{\partial \theta} =
    \sum \left[ \frac{\partial \hat{y}_i(\theta)}{\partial \theta} \mathrm{exp}(\hat{y}_i(\theta) - Q) \right]

Now that we have our numerically stable expressions for :math:`\hat{y}` and :math:`\frac{\partial \hat{y}}{\partial \theta}`, we can move on to implementing this in the code.

Code
----

To implement this in the code, we'll need to write two classes: ``MaxCombinationFunc``, which subclasses the ``torch.autograd.Function`` class and handles all the logic for computing and returning gradients, and ``MaxCombination``, which subclasses the abstract :py:class:`mtenn.combination.Combination` class and wraps the ``MaxCombinationFunc`` class into a ``torch.Module``.

``MaxCombinationFunc``
^^^^^^^^^^^^^^^^^^^^^^^

To subclass ``torch.autograd.Function``, ``MaxCombinationFunc`` needs to implement three ``@staticmethod`` functions: ``forward``, ``setup_context``, and ``backward``.

.. code-block:: python

    from mtenn.combination import Combination
    import torch

    class MaxCombinationFunc(torch.autograd.Function):

        @staticmethod
        def forward(pred_list, grad_dict, param_names, *model_params):
            ...

        @staticmethod
        def setup_context(ctx, inputs, output):
            ...

        @staticmethod
        def backward(ctx, grad_output):
            ...


``forward``
"""""""""""

The ``forward`` method should be familiar, and is only responsible for applying the math to combine the single-pose predictions into an overall multi-pose prediction.
The only thing that may be a bit strange here is the extra inputs to the function.
These are an artifact of us needing these inputs in ``setup_context``, and we don't need them in ``forward``.

.. code-block:: python

    @staticmethod
    def forward(pred_list, grad_dict, param_names, *model_params):
        """
        pred_list: List[torch.Tensor]
            List of delta G predictions to be combined using LSE
        grad_dict: dict[str, List[torch.Tensor]]
            Dict mapping from parameter name to list of gradient
            (not used in this function)
        param_names: List[str]
            List of parameter names (not used in this function)
        model_params: torch.Tensor
            Actual parameters that we'll return the gradients for. Each param
            should be passed individually for the backward pass to work right.
            (not used in this function)
        """
        # Overall multi-pose prediction is given by simply taking the LSE of all preds
        final_pred = torch.logsumexp(torch.stack(pred_list).flatten(), dim=0).detach()

        return final_pred

``setup_context``
"""""""""""""""""

The ``setup_context`` method is responsible for handling all the logic of saving information that will be used in the backward pass.
In our case, we will want to store the single-pose predictions and their gradients (all of which have already been calculated prior to the ``Combination`` block).

The logic and reasoning here deserve some special attention, as it's a bit convoluted.
The overall goal of going through this whole process is that we want to ``pytorch`` to automatically attach our pre-computed gradients to the appropriate tensors when we ultimately call ``loss.backward()`` on the loss value calculated from our multi-pose prediction.
To that end, the parameters themselves will need to be passed to our ``Combination`` block so that we can return the gradients for them in ``backward``.

In addition to the actual model parameter tensors, we also need to pass some extra information along to ``backward``.
Obviously we will need the list of single-pose predictions (``pred_list``) and the gradients of those predictions (``grad_dict``).
As the name implies, ``pred_list`` is a ``list`` of the single-pose predictions, stored as tensors.
``grad_dict`` is a ``dict`` that maps from a model parameter name to a list of gradients for that parameter.
The gradient at index :math:`i` in each list corresponds to the gradient of the :math:`i` th prediction wrt that paramter.
The set of ``grad_dict.keys()`` must be equal to the set of ``param_names``.
``param_names`` is a ``list`` of model parameter names that corresponds directly to the parameter tensors that are passed, ie the parameter in ``model.state_dict()`` that is accessed by the :math:`i` th name in ``param_names`` should be the :math:`i` th tensor in ``model_params``.

.. code-block:: python

    @staticmethod
    def setup_context(ctx, inputs, output):
        """
        ctx is the context manager that will store values for use in the backward pass.

        The contents of inputs should be:

        pred_list: List[torch.Tensor]
            List of delta G predictions to be combined using LSE
        grad_dict: dict[str, List[torch.Tensor]]
            Dict mapping from parameter name to list of gradients
        param_names: List[str]
            List of parameter names
        *model_params: torch.Tensor
            Actual parameters that we'll return the gradients for. Each param
            should be passed individually for the backward pass to work right.

        The contents of output will be everything that was returned by forward. In our
        case, we don't need that value as an intermediate so we can just ignore it.
        """

        # Split up inputs
        pred_list, grad_dict, param_names, *model_params = inputs

        # Decompose grad_dict into a list of parameter names and a flattened list of
        #  per-prediction gradients (that correspond 1:1 to each other)
        grad_dict_keys, grad_dict_tensors = Combination.split_grad_dict(grad_dict)

        # Non-Tensor values can be saved for backward by assigning directly to the
        #  context object
        ctx.grad_dict_keys = grad_dict_keys
        ctx.param_names = param_names

        # Tensor values must be saved using the save_for_backward method
        # Saving:
        #  * Predictions (1 tensor)
        #  * Grad tensors (N params * M poses tensors)
        #  * Model param tensors (N params tensors)
        ctx.save_for_backward(
            torch.stack(pred_list).flatten(),
            *grad_dict_tensors,
            *model_params,
        )

``backward``
""""""""""""

The ``backward`` method is where we actually do the computations that we solved for in the :ref:`new-comb-math` section.
Code-wise, this is fairly simple.
All we need to do is reconstruct the ``grad_dict`` that we flattened in ``setup_context``, do the math, and return the appropriate gradients at the end.

The ``grad_output`` value in the function inputs contains the gradient accumulated in the value returned from forward up to this point in the computation graph.
In our case, this should just be a scalar value as the loss should be calculated directly on the multi-pose prediction returned from ``forward``.

.. code-block:: python

    @staticmethod
    def backward(ctx, grad_output):
        """
        ctx is the same context manager from setup_context.
        """

        # Unpack saved tensors
        # We know the first tensor is the list of single-pose predictions, so we can
        #  pop that out first
        preds, *other_tensors = ctx.saved_tensors

        # other_tensors is the list of the flattened grad_dict tensors + the model
        #  parameter tensors that were passed in
        # We know that there are exactly as many grad_dict tensors as there are
        #  grad_dict_keys, so we can take those out as well
        # We don't actually use the model_params tensors, they just need to be passed
        #  so that pytorch knows to assign gradients to them
        grad_dict_tensors = other_tensors[: len(ctx.grad_dict_keys)]

        # Reconstruct the dict that we previously flattened
        grad_dict = Combination.join_grad_dict(ctx.grad_dict_keys, grad_dict_tensors)

        # Calculate our numericall stable substitution value
        Q = torch.logsumexp(preds.detach(), dim=0)

        # Calculate final gradients for each parameter
        final_grads = {}
        for n, grad_list in grad_dict.items():
            final_grads[n] = (
                torch.stack(
                    [
                        grad * (pred - Q).exp()
                        for grad, pred in zip(grad_list, preds)
                    ],
                    axis=-1,
                )
                .detach()
                .sum(axis=-1)
            )

        # Multiply gradients by scalar in grad_output
        for grad in final_grads.values():
            grad *= grad_output

        # Need to return a gradient for each value that was passed in inputs, which will
        #  be the calculated gradients for each of the model_params, and None for
        #  everything else
        return_vals = [None] * 3 + [final_grads[n] for n in ctx.param_names]
        return tuple(return_vals)

``MaxCombination``
^^^^^^^^^^^^^^^^^^^^^^^

The implementation for the ``MaxCombination`` class is fairly simple.
In order to subclass the :py:class:`mtenn.combination.Combination` abstract class, it only needs to implement the ``forward`` method, which should take as inputs all of the inputs that we discussed above in the ``MaxCombinationFunc.setup_context`` function.
The only thing we need to do in this ``forward`` method is call the ``MaxCombinationFunc.apply`` function, which is implemented in ``torch.autograd.Function``, and handles the calling of the ``MaxCombinationFunc.forward``, ``MaxCombinationFunc.setup_context``, and ``MaxCombinationFunc.backward`` functions.

.. code-block:: python

    class MaxCombination(Combination):
        """
        Approximate max of the predictions using the LogSumExp function for smoothness.
        """

        def __init__(self):
            super(MaxCombination, self).__init__()

        def forward(self, pred_list, grad_dict, param_names, *model_params):
            return MaxCombinationFunc.apply(
                pred_list, grad_dict, param_names, *model_params
            )
