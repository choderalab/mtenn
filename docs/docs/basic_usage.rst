Basic Usage
===========

An ``mtenn.Model`` object is build using a config object, and then can be used as a standard ``pytorch`` model.
More details on the config system are in the :py:mod:`mtenn.combination` docs, and more details on the expected model inputs are in :doc:`model`.
Below, we detail a basic example of building a default Graph Attention model and using it to make a prediction on a SMILES string.

.. code-block:: python

    from mtenn.config import GATModelConfig
    import rdkit.Chem as Chem
    import torch

    # Build model with GAT defaults
    model = GATModelConfig().build()

    # Build mol
    smiles = "CCCC"
    mol = Chem.MolFromSmiles(smiles)

    # Get atomic numbers and bond indices (both directions)
    atomic_nums = [a.GetAtomicNum() for a in mol.GetAtoms()]
    bond_idxs = [
        atom_pair
        for bond in mol.GetBonds()
        for atom_pair in (
            (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
            (bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()),
        )
    ]
    # Add self bonds
    bond_idxs += [(a.GetIdx(), a.GetIdx()) for a in mol.GetAtoms()]

    # Encode atomic numbers as one-hot, assume max num of 100
    node_feats = torch.nn.functional.one_hot(
        torch.tensor(atomic_nums), num_classes=100
    ).to(dtype=torch.float)
    # Format bonds in correct shape
    edge_index = torch.tensor(bond_idxs).t()

    # Make a prediction
    pred, _  = model({"x": node_feats, "edge_index": edge_index})

