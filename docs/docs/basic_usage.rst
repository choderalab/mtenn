Basic Usage
===========

An ``mtenn.Model`` object is build using a config object, and then can be used as a standard ``pytorch`` model.
More details on the config system are in :doc:`config`, and more details on the expected model inputs are in :doc:`model`.
Below, we detail a basic example of building a default Graph Attention model and using it to make a prediction on a SMILES string.

.. code-block:: python

    from dgllife.utils import CanonicalAtomFeaturizer, SMILESToBigraph
    from mtenn.config import GATModelConfig

    # Build model with GAT defaults
    model = GATModelConfig().build()

    # Build graph from SMILES
    smiles = "CCCC"
    g = SMILESToBigraph(
        add_self_loop=True,
        node_featurizer=CanonicalAtomFeaturizer(),
    )(smiles)

    # Make a prediction
    pred, _ = model({"g": g})

