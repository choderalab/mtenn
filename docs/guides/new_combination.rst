.. _new-combination-guide:

Implementing a new Combination
==============================

This guide will assume that you've already read the :doc:`../docs/combination` docs page, so head there first if you haven't.
In this guide, we will walk through both the math and the software engineering that was done to implement :py:class:`MaxCombination <mtenn.combination.MaxCombination>`.
These steps should hopefully be illustrative enough to enable anyone to implement their own ``Combination`` method.

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

To implement this in the code, we'll need to write two classes: ``_MaxCombinationFunc``, which subclasses the ``torch.autograd.Function`` class and handles all the logic for computing and returning gradients, and ``MaxCombination``, which subclasses the abstract :py:class:`mtenn.combination.Combination` class and wraps the ``_MaxCombinationFunc`` class into a ``torch.Module``.

``_MaxCombinationFunc``
^^^^^^^^^^^^^^^^^^^^^^^

To subclass ``torch.autograd.Function``, ``_MaxCombinationFunc`` needs to implement three functions: ``setup_context``, ``forward``, and ``backward``.

``forward``
"""""""""""

The ``forward`` method should be familiar, and is only responsible for applying the math to combine the single-pose predictions into an overall multi-pose prediction.
The only thing that may be a bit strange here is the extra inputs to the function.
These are an artifact of all the inputs in ``setup_context`` also being passed to ``forward``.
We don't need these values here, but they are necessary in the ``setup_context`` method.
Note also that ``forward`` needs to have the ``@staticmethod`` decorator for everything to work properly.

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
