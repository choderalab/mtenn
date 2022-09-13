import h5py
from torch.utils.data import Dataset


class SPICELoader(Dataset):
    def __init__(self, hdf5_file):
        super(SPICELoader, self).__init__()

        self.file = h5py.File(hdf5_file)
        self.leaf_names = self._get_leaves()

    def __iter__(self):
        for l in self.leaf_names:
            yield self[l]

    def __getitem__(self, idx):
        if type(idx) is int:
            l = self.leaf_names[idx]
            g = self.file[l]
        elif type(idx) is str:
            l = idx
            g = self.file[idx]
        else:
            raise TypeError(
                f"Unspported indexing type {type(idx)} (must be int or str)"
            )

        return {
            "name": l,
            "smiles": g["smiles"][0].decode(),
            "z": g["atomic_numbers"][:],
            "pos": g["conformations"][:],
            "formation_energy": g["formation_energy"][:],
            "grad": g["dft_total_gradient"][:],
        }

    def __len__(self):
        return len(self.leaf_names)

    def _get_leaves(self):
        leaf_list = []

        def visit_func(name, obj):
            if (type(obj) is h5py._hl.group.Group) and (
                "atomic_numbers" in obj.keys()
            ):
                leaf_list.append(name)

        self.file.visititems(visit_func)
        return leaf_list
