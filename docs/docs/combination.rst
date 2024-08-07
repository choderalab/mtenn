.. _comb-docs-page:

Combination
===========

The naive way of implementing a multi-pose model would be to simply loop over all the input structures, pass the predictions to a ``Combination`` block, and then use that final prediction to calculate the loss and perform backpropagation (in training).
However, this implementation requires keeping the full computation graph for each input structure in GPU memory.
This may be fine for smaller structures, but when working with all-heavy-atom representations of full protein-ligand structures, these graphs quickly overrun the memory on a standard GPU.
To get around this limitation, we rework the internals of the ``GroupedModel`` class as follows:

#. For each input pose, make a prediction and calculate the gradient of the prediction with respect to the model parameters (using ``prediction.backward()``), and store both the prediction and the the gradients for each parameter
#. Pass the predictions and gradients to the ``Combination`` block, which combines the predictions into a final prediction and stores all the values to be used during the backward pass

By decoupling the backward pass of the underlying single-pose model from the backward pass of the overall ``GroupedModel``, we no longer need to keep track of the entire computation graph for all of the input poses.
To accomplish this effect, the ``Combination`` blocks use custom ``torch.autograd.Function`` s, which handle the storing and combining of the gradients for individual poses, to then be returned when ``backward`` is called on the overall model prediction.
For more details on how exactly this is done, see our guide on :ref:`new-combination-guide`.

The general mathematical theory behind this approach is fairly simple.
The prediction for each pose is generated by the same single-pose model (:math:`f(\theta)`)

.. math::

    \hat{y}_i = f( \text{X}_i, \theta )

and the final prediction for this compound is found by applying the combination function (:math:`h`) to this set of individual predictions:

.. math::

    \hat{y}(\theta) = h ( \hat{y}_1(\theta), ..., \hat{y}_n(\theta) )

We then calculate the loss of our prediction compared to a target value

.. math::

    \text{loss} = L ( \hat{y}(\theta), y )

and backprop is performed by calcuation the gradient of that loss wrt the model parameters:

.. math::

    \frac{\partial \text{loss}}{\partial \theta} = \frac{\partial L}{\partial \hat{y}} \frac{\partial \hat{y}}{\partial \theta}

The :math:`\frac{\partial L}{\partial \hat{y}}` term can be calculated automatically using the ``pytorch.autograd`` capabilities.
However, because we've decoupled the single-pose model predictions from the overall multi-pose prediction, we must manually account for the relation between the :math:`\frac{\partial \hat{y}}{\partial \theta}` term and the individual gradients that we calculated during the forward pass (:math:`\frac{\partial \hat{y}_i}{\partial \theta}`).
Arbitrarily, this will be some function (:math:`g`) that depends on the individual predictions and their gradients:

.. math::

    \frac{\partial \hat{y}}{\partial \theta} = \frac{\partial h}{\partial \theta} =
    g( \hat{y}_1, ..., \hat{y}_n, \frac{\partial \hat{y}_1}{\partial \theta}, ..., \frac{\partial \hat{y}_n}{\partial \theta} )

In practice, this function :math:`g` will need to be analytically determined and manually implemented within the ``Combination`` block (see :ref:`the guide <new-combination-guide>` for more practical information).

.. _implemented-combs:

Math for Implemented Combinations
----------------------------------

Below, we detail the math required for appropriately combining gradients.
This math is used in the ``backward`` pass in the various ``Combination`` classes.

.. _imp-comb-loss-fn:

Loss Functions
^^^^^^^^^^^^^^

We anticipate these ``Combination`` methods being used with a linear combination of two types of  loss functions:

    * Loss based on the final combined prediction (ie :math:`L = f(\Delta \text{G} (\theta))`)

    * Loss based on a linear combination of the per-pose predictions (ie :math:`L = f(\Delta \text{G}_1 (\theta), \Delta \text{G}_2 (\theta), ...)`)

Ultimately for backprop we need to return the gradients of the loss wrt each model parameter.
The gradients for each of these types of losses is given below.

Combined Prediction
"""""""""""""""""""

.. math::
    :label: comb-grad

    \frac{\partial L}{\partial \theta} =
    \frac{\partial L}{\partial \Delta \text{G}}
    \frac{\partial \Delta \text{G}}{\partial \theta}

The :math:`\frac{\partial L}{\partial \Delta \text{G}}` part of this equation will be a scalar that is calculated automatically by ``pytorch`` and fed to our ``Combination`` class.
The :math:`\frac{\partial \Delta \text{G}}{\partial \theta}` parts will be computed internally.

Per-Pose Prediction
"""""""""""""""""""

Because we assume this loss is based on a linear combination of the individual :math:`\Delta \text{G}_i` predictions, we can decompose the loss as:

.. math::
    :label: pose-grad

    \frac{\partial L}{\partial \theta} =
    \sum_{i=1}^N
    \frac{\partial L}{\partial \Delta \text{G}_i}
    \frac{\partial \Delta \text{G}_i}{\partial \theta}

As before, the :math:`\frac{\partial L}{\partial \Delta \text{G}_i}` parts of this equation will be scalars calculated automatically by ``pytorch`` and fed to our ``Combination`` class, and the :math:`\frac{\partial \Delta \text{G}}{\partial \theta}` parts will be computed internally.

.. _mean-comb-imp:

Mean Combination
^^^^^^^^^^^^^^^^

This is mostly included as an example, but it can be illustrative.

.. math::
    :label: mean-comb-pred

    \Delta \text{G}(\theta) = \frac{1}{N} \sum_{i=1}^{N} \Delta \text{G}_i (\theta)

.. math::
    :label: mean-comb-grad

    \frac{\partial \Delta \text{G}(\theta)}{\partial \theta} = \frac{1}{N} \sum_{i=1}^{N} \frac{\partial \Delta \text{G}_i (\theta)}{\partial \theta}

.. _max-comb-imp:

Max Combination
^^^^^^^^^^^^^^^

This will likely be the more useful of the currently implemented ``Combination`` implementations.
In the below equations, we define the following variables:

    * :math:`n` : A sign multiplier taking the value of :math:`-1` if we are taking the min value (generally the case if the inputs are :math:`\Delta \text{G}` values) or :math:`1` if we are taking the max
    * :math:`t` : A scaling value that will bring the final combined value closer to the actual value of the max/min of the input values (see `here <https://en.wikipedia.org/wiki/LogSumExp#Properties>`_ for more details).
      Setting :math:`t = 1` reduces this operation to the LogSumExp operation

.. math::
    :label: max-comb-pred

    \Delta \text{G}(\theta) = n \frac{1}{t} \text{ln} \sum_{i=1}^N \text{exp} (n t \Delta \text{G}_i (\theta))

We define a a constant :math:`Q` for simplicity as well as for numerical stability:

.. math::
    :label: max-comb-q

    Q = \text{ln} \sum_{i=1}^N \text{exp} (n t \Delta \text{G}_i (\theta))

.. math::
    :label: max-comb-grad-initial

    \frac{\partial \Delta \text{G}(\theta)}{\partial \theta} =
    n^2
    \frac{1}{\sum_{i=1}^N \text{exp} (n t \Delta \text{G}_i (\theta))}
    \sum_{i=1}^N \left[
    \frac{\partial \Delta \text{G}_i (\theta)}{\partial \theta} \text{exp} (n t \Delta \text{G}_i (\theta))
    \right]

Substituting in :math:`Q`:

.. math::
    :label: max-comb-grad-sub

    \frac{\partial \Delta \text{G}(\theta)}{\partial \theta} =
    \frac{1}{\text{exp}(Q)}
    \sum_{i=1}^N \left[
    \text{exp} \left( n t \Delta \text{G}_i (\theta) \right) \frac{\partial \Delta \text{G}_i (\theta)}{\partial \theta}
    \right]

.. math::
    :label: max-comb-grad-final

    \frac{\partial \Delta \text{G}(\theta)}{\partial \theta} =
    \sum_{i=1}^N \left[
    \text{exp} \left( n t \Delta \text{G}_i (\theta) - Q \right) \frac{\partial \Delta \text{G}_i (\theta)}{\partial \theta}
    \right]
