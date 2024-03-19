Model
=====

This document describes the classes in ``mtenn.model``, and their substituent parts.

Model Blocks
------------

Each class in ``mtenn.model`` is comprised of at least one of the following blocks:

* :ref:`representation-block`
* :ref:`strategy-block`
* :ref:`readout-block`

.. _representation-block:

Representation
^^^^^^^^^^^^^^

The ``Representation`` block (``mtenn.representation``) is responsible for taking an input structure and learning an n-dimensional vector embedding.
In practice, this block is a thin wrapper around an existing model architecture, potentially with some ad hoc manipulation done to ensure the output is a vector rather than a single scalar value.

For more information on the models that are currently implemented in ``mtenn``, see the :ref:`current-models` section.

For information on adding new models into the ``mtenn`` framework, see the guides on :ref:`adding an installable model <new-installable-model-guide>` and :ref:`adding a non-installable model <new-non-installable-model-guide>`.


.. _strategy-block:

Strategy
^^^^^^^^

The ``Strategy`` block (``mtenn.strategy``) is responsible for taking any number of vectors, each output from a ``Representation`` block, and combining them into a :math:`\Delta g_{\mathrm{bind}}` prediction in kT units.

Currently, the following ``Strategy`` blocks are implemented in ``mtenn``:

* :py:class:`mtenn.strategy.DeltaStrategy`
* :py:class:`mtenn.strategy.ConcatStrategy`
* :py:class:`mtenn.strategy.ComplexOnlyStrategy`

.. _readout-block:

Readout
^^^^^^^

Single-Pose Models
------------------

This section is a description of the ``mtenn.model.Model`` class (referred to as ``Model`` from here), which makes a prediction on a single input conformation.
The general data flow through a ``Model`` object is as depicted in the below diagram:

.. image:: /static/mtenn_model_diagram.png

In text form this is:

#. The protein-ligand complex structure is passed to the ``Model``
#. Internally, the ``Model`` breaks the structure into 3 sub-structures: the full complex, just the protein, and just the ligand
#. Each of these sub-strucuures is individually passed to the ``Representation`` block to generate a total of 3 vector representations
#. All 3 representations are passed to the ``Strategy`` block, where they are combined into a :math:`\Delta g_{\mathrm{bind}}` prediction in implicit kT units
#. (optional) The :math:`\Delta g_{\mathrm{bind}}` prediction is passed to the ``Readout`` block, where it is converted  into whatever the final units are

Multi-Pose Models
-----------------

Ligand-Only Models
------------------

.. _current-models:

Currently Implemented Models
----------------------------
