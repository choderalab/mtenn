import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from mtenn.conversion_utils.schnet import SchNet
from mtenn.conversion_utils.e3nn import E3NN
import yaml
import ast
import argparse
import os
import pickle as pkl
from covid_moonshot_ml.utils import find_most_recent, plot_loss
from e3nn import o3
from torch_cluster import radius_graph
from collections import Counter
from covid_moonshot_ml.data.dataset import DockedDataset
from glob import glob
import re
import json
from covid_moonshot_ml.schema import ExperimentalCompoundDataUpdate
from torch_geometric.datasets import QM9
from torch_geometric.nn import SchNet as PyGSchNet


def add_one_hot_encodings(ds):
    """
    Add 100-length one-hot encoding of the atomic number for each entry in ds.
    Needed to match the expected format for e3nn model.
    Parameters
    ----------
    ds : data.dataset.DockedDataset
        Dataset to add encodings to
    Returns
    -------
    data.dataset.DockedDataset
        Dataset with one-hot encodings
    """
    for comp in ds:
        temp = torch.tensor(comp["z"] - 1)
        comp["x"] = torch.nn.functional.one_hot(
            temp.to(torch.int64), 100
        ).float()

    return ds


def calc_e3nn_model_info(ds, r):
    """
    Calculate parameters to use in creation of an e3nn model.
    Parameters
    ----------
    ds : data.dataset.DockedDataset
        Dataset of structures to use to calculate the parameters
    r : float
        Cutoff to use for neighbor atom calculations
    Returns
    -------
    int
        Number of unique atom types found in `ds`
    float
        Average number of neighbors for each node across all of `ds`
    int
        Rounded average number of nodes per structure in `ds`
    """
    num_neighbors = []
    num_nodes = []
    unique_atom_types = set()
    for pose in ds:
        for conf in pose["pos"]:
            temp = torch.tensor(conf)
            edge_src, edge_dst = radius_graph(x=temp, r=r)
            num_neighbors.extend(Counter(edge_src.numpy()).values())
            num_nodes.append(temp.shape[0])
            unique_atom_types.update(pose["z"].tolist())

    return (
        len(unique_atom_types),
        np.mean(num_neighbors),
        round(np.mean(num_nodes)),
    )


def get_e3nn_kwargs(ds, cutoff):
    """
    Add 100-length one-hot encoding of the atomic number for each entry in ds.
    Needed to match the expected format for e3nn model.
    Parameters
    ----------
    ds : data.dataset.DockedDataset
        Dataset to add encodings to
    cutoff: float
        Cutoff to use for neighbor atom calculations
    Returns
    -------
    dictionary
        keyword arugments for setting up the e3nn model
    """
    model_params = calc_e3nn_model_info(ds, cutoff)

    model_kwargs = {
        "irreps_in": f"{100}x0e",
        "irreps_hidden": [
            (mul, (l, p)) for l, mul in enumerate([10, 3]) for p in [-1, 1]
        ],
        "irreps_out": "1x0e",
        "irreps_node_attr": None,
        "irreps_edge_attr": o3.Irreps.spherical_harmonics(1),
        "layers": 3,
        "max_radius": 3.5,
        "number_of_basis": 10,
        "radial_layers": 1,
        "radial_neurons": 128,
        "num_neighbors": model_params[1],
        "num_nodes": model_params[2],
        "reduce_output": True,
    }

    return model_kwargs


def get_params(fn):
    """
    Reads yaml file that contains training and model paramaters
    Parameters
    ----------
    fn : String
        Filename for yaml files
    Returns
    -------
    dictionary
        Model  and training paramters
    """
    with open(fn, "r") as c:
        config = yaml.safe_load(c)

    # For strings that yaml doesn't parse (e.g. None)
    for key, val in config.items():
        if type(val) is str:
            try:
                config[key] = ast.literal_eval(val)
            except (ValueError, SyntaxError):
                pass
    params = {
        "dataset": config["dataset"],
        "train_split": config["train_split"],
        "val_split": config["val_split"],
        "random_seed": config["random_seed"],
        "n_epochs": config["n_epochs"],
        "batch_size": config["batch_size"],
        "model": config["model"],
        "cont": config["cont"],
        "model_o": config["model_o"],
        "model_in": config["model_in"],
        "plot_o": config["plot_o"],
        "dataset_labels": config["dataset_labels"],
        "qm9": config["qm9"],
    }

    return params


def get_args():
    """
    Gets filename for training parameters
    """

    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "-config",
        required=True,
        help="Filepath to parameter configuration file",
    )

    return parser.parse_args()


def train_loop(dataloader, model, loss_func, optimizer, device):
    """
    Training loop
    """
    batch_losses = []
    for batch in dataloader:
        optimizer.zero_grad()
        preds = []
        targets = []
        for d in batch:
            pos = torch.tensor(d["pos"], device=device)
            x = torch.tensor(d["x"], device=device)

            lig_index = d["lig"]
            prot_index = np.invert(d["lig"])
            x_lig = x[lig_index]
            pos_lig = pos[lig_index]
            x_prot = x[prot_index]
            pos_prot = pos[prot_index]

            rep_comp = {"x": x, "pos": pos}
            rep_lig = {"x": x_lig, "pos": pos_lig}
            rep_prot = {"x": x_prot, "pos": pos_prot}
            pred = model(rep_comp, rep_prot, rep_lig)

            x, pos = [], []
            preds.append(pred.reshape((1,)))
            targets.append(
                torch.tensor(d["energy"], device=device).reshape((1,))
            )
        loss = loss_func(torch.stack(preds), torch.stack(targets))
        loss.backward()
        optimizer.step()
        preds, targets = [], []
        batch_losses.append(loss.item())

    return batch_losses


def val_loop(dataloader, model, loss_func, device):
    """
    Validation loop
    """
    with torch.no_grad():
        batch_losses = []
        for batch in dataloader:
            preds = []
            targets = []
            for d in batch:
                pos = torch.tensor(d["pos"], device=device)
                x = torch.tensor(d["x"], device=device)

                lig_index = d["lig"]
                prot_index = np.invert(d["lig"])
                x_lig = x[lig_index]
                pos_lig = pos[lig_index]
                x_prot = x[prot_index]
                pos_prot = pos[prot_index]

                rep_comp = {"x": x, "pos": pos}
                rep_lig = {"x": x_lig, "pos": pos_lig}
                rep_prot = {"x": x_prot, "pos": pos_prot}
                pred = model(rep_comp, rep_prot, rep_lig)

                x, pos = [], []
                preds.append(pred.reshape((1,)))
                targets.append(
                    torch.tensor(d["energy"], device=device).reshape((1,))
                )
            loss = loss_func(torch.stack(preds), torch.stack(targets))
            preds = []
            targets = []
            batch_losses.append(loss.item())
        return batch_losses


def train(
    model,
    train_dataloader,
    val_dataloader,
    n_epochs,
    loss_func,
    device,
    save_file=None,
    start_epoch=0,
    train_loss=[],
    test_loss=[],
):
    """
    Loop for training, validating, and saving model results
    """
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch_idx in range(start_epoch, n_epochs):
        print(
            "Epoch " + str(epoch_idx + 1) + " of " + str(n_epochs), flush=True
        )
        print("training", flush=True)
        train_batch_losses = train_loop(
            train_dataloader, model, loss_func, optimizer, device
        )
        print(np.mean(train_batch_losses), flush=True)
        print("validating", flush=True)
        val_batch_losses = val_loop(val_dataloader, model, loss_func, device)
        print(np.mean(val_batch_losses), flush=True)

        train_loss.append(np.mean(train_batch_losses))
        test_loss.append(np.mean(val_batch_losses))

        if save_file is None:
            continue
        elif os.path.isdir(save_file):
            torch.save(model.state_dict(), f"{save_file}/{epoch_idx}.th")
            pkl.dump(
                np.vstack(train_loss), open(f"{save_file}/train_err.pkl", "wb")
            )
            pkl.dump(
                np.vstack(test_loss), open(f"{save_file}/test_err.pkl", "wb")
            )
        elif "{}" in save_file:
            torch.save(model.state_dict(), save_file.format(epoch_idx))
        else:
            torch.save(model.state_dict(), save_file)

    return model, np.vstack(train_loss), np.vstack(test_loss)


def load_affinities(fn, achiral=True):
    """
    Load binding affinities from JSON file of
    schema.ExperimentalCompoundDataUpdate.
    Parameters
    ----------
    fn : str
        Path to JSON file
    achiral : bool, default=True
        Whether to only take achiral molecules
    Returns
    -------
    dict[str->float]
        Dictionary mapping coumpound id to experimental pIC50 value
    """
    ## Load all compounds with experimental data and filter to only achiral
    ##  molecules (to start)
    exp_compounds = ExperimentalCompoundDataUpdate(
        **json.load(open(fn, "r"))
    ).compounds
    exp_compounds = [c for c in exp_compounds if c.achiral == achiral]

    affinity_dict = {
        c.compound_id: c.experimental_data["pIC50"]
        for c in exp_compounds
        if "pIC50" in c.experimental_data
    }

    return affinity_dict


def merge_labels_and_data(affinities, ds):
    """
    Put dataset in usable format for training loop
    Parameters
    ----------
    affinities : dictionary
        Maps compound ID to pIC50
    ds : data.dataset.DockedDataset
        Dataset with pose information
    Returns
    -------
    array
        Dataset with energy and pose
    """
    final_ds = []
    for (_, compound_id), pose in ds:
        comp_dict = pose.copy()
        comp_dict["energy"] = affinities[compound_id]
        final_ds.append(comp_dict)

    return final_ds


def init_data(params):
    """
    Initialize datasets
    Parameters
    ----------
    params : dictionary
        Dictionary containing training and model info
    Returns
    -------
    torch.utils.data.Dataloader
        Training data
    torch.utils.data.Dataloader
        Validation data
    torch.utils.data.Dataloader
        Testing data
    """

    all_fns = glob(f'{params["dataset"]}/*complex.pdb')
    re_pat = r"(Mpro-P[0-9]{4}_0[AB]).*?([A-Z]{3}-[A-Z]{3}-.*?)_complex.pdb"
    compounds = [re.search(re_pat, fn).groups() for fn in all_fns]

    ## Load the experimental affinities
    exp_affinities = load_affinities(params["dataset_labels"])

    ## Trim docked structures and filenames to remove compounds that don't have
    ##  experimental data
    all_fns, compounds = zip(
        *[o for o in zip(all_fns, compounds) if o[1][1] in exp_affinities]
    )

    ds = DockedDataset(all_fns, compounds)

    total_ds = merge_labels_and_data(exp_affinities, ds)

    total_ds = add_one_hot_encodings(total_ds)

    n_train = int(len(total_ds) * params["train_split"])
    n_val = int(len(total_ds) * params["val_split"])
    n_test = len(total_ds) - n_train - n_val

    batch_size = params["batch_size"]

    train_data, val_data, test_data = random_split(
        total_ds,
        [n_train, n_val, n_test],
        torch.Generator().manual_seed(params["random_seed"]),
    )

    def collate_fn(batch):
        return batch

    train_dataloader = DataLoader(
        train_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    val_dataloader = DataLoader(
        val_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    return train_dataloader, val_dataloader, train_data


def main():
    args = get_args()
    params = get_params(args.config)

    loss_func = torch.nn.MSELoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_dataloader, val_dataloader, spl_train = init_data(params)

    model_kwargs = get_e3nn_kwargs(spl_train, 3.5)
    m = E3NN(model_kwargs)
    wts_fn = find_most_recent(params["model_in"])[1]
    m.load_state_dict(torch.load(wts_fn))
    model = E3NN.get_model(model=m, strategy="delta hartree")

    ## Load model weights as necessary
    if params["cont"]:
        print("continuing")
        start_epoch, wts_fn = find_most_recent(params["model_o"])
        # print(start_epoch)
        model.load_state_dict(torch.load(wts_fn))

        ## Load error dicts
        if os.path.isfile(f'{params["model_o"]}/train_err.pkl'):
            train_loss = pkl.load(
                open(f'{params["model_o"]}/train_err.pkl', "rb")
            ).tolist()
        else:
            train_loss = []
        if os.path.isfile(f'{params["model_o"]}/test_err.pkl'):
            test_loss = pkl.load(
                open(f'{params["model_o"]}/test_err.pkl', "rb")
            ).tolist()
        else:
            test_loss = []

        ## Need to add 1 to start_epoch bc the found idx is the last epoch
        ##  successfully trained, not the one we want to start at
        start_epoch += 1
    else:
        start_epoch = 0
        train_loss = []
        test_loss = []

    print("Beginning training", flush=True)

    model, train_loss, test_loss = train(
        model,
        train_dataloader,
        val_dataloader,
        params["n_epochs"],
        loss_func,
        device,
        params["model_o"],
        start_epoch,
        train_loss,
        test_loss,
    )

    if params["plot_o"] is not None:
        plot_loss(
            train_loss.mean(axis=1), test_loss.mean(axis=1), params["plot_o"]
        )


if __name__ == "__main__":
    main()
