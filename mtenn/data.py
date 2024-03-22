"""
Thin data wrappers for conveninece and uniformity.
"""

from collections import namedtuple

"""
Data representation for inputs to a 2D model. The only information required here is the
2D representation, whatever shape that may take (graph, fingerprint, etc). An additional
optional field is provided for labeling which components of the representation belong
to, eg if your representation is a concatenated bit vector of protein + ligand
representations.
"""
NonStructData = namedtuple(
    "NonStructData", ["representation", "component_idx"], defaults=(None,)
)

"""
Data representation for inputs to a 3D model. This data representation requires the
coordinates and atomic numbers of each atom. There is also the same optional component
labeling that is available in the NonStructData class, as well as an additional field
that can be filled with additional features to attach to each atom.
A typical setup for this will be:
  * pos: (n, 3) tensor, giving 3D coordinates for each of n atoms
  * z: (n, ) tensor, giving integer atomic numbers for each of n atoms
  * component_idx: (n, ) tensor, giving integer labels for which component (likely
    protein and ligand) each of the n atoms belongs to
  * extra_feats: (n, m) tensor, giving an m-dimensional feature vector for each of the
    n atoms
"""
StructData = namedtuple(
    "StructData", ["pos", "z", "component_idx", "extra_feats"], defaults=(None, None)
)
