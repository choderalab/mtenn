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

def get_params(fn):
    '''
    Reads yaml file that contains training and model paramaters
    Parameters
    ----------
    fn : String
        Filename for yaml files
    Returns
    -------
    dictionary
        Model  and training paramters
    '''
    with open(fn, 'r') as c:
        config = yaml.safe_load(c)
    
    # For strings that yaml doesn't parse (e.g. None)
    for key, val in config.items():
        if type(val) is str:
            try:
                config[key] = ast.literal_eval(val)
            except (ValueError, SyntaxError):
                pass
    params = {
        "dataset" : config["dataset"],
        "train_split" : config["train_split"],
        "val_split" : config["val_split"],
        "random_seed" : config["random_seed"],
        "n_epochs" : config["n_epochs"],
        "batch_size" : config["batch_size"],
        "model" : config["model"],
        "cont" : config["cont"],
        "model_o" : config["model_o"],
        "model_in" : config["model_in"],
        "plot_o" : config["plot_o"],
        "dataset_labels" : config["dataset_labels"],
        "qm9" : config["qm9"],
    }
    
    return params


def get_args():
    '''
    Gets filename for training parameters
    '''
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-config', required=True,
        help='Filepath to parameter configuration file')

    return(parser.parse_args())

def train_loop(dataloader, model, loss_func, optimizer, device):
    '''
    Training loop
    '''
    batch_losses = []
    for batch in dataloader:
        optimizer.zero_grad()
        preds = []
        targets = []
        for d in batch:
            z = torch.tensor(d['z'], dtype=torch.long, device=device)
            pos = torch.tensor(d['pos'], device=device)
            lig_index=d['lig']
            prot_index = np.invert(d['lig'])
            z_lig = z[lig_index]
            pos_lig = pos[lig_index]
            z_prot = z[prot_index]
            pos_prot = pos[prot_index]

            rep_comp= (z,pos)
            rep_lig= (z_lig, pos_lig)
            rep_prot= (z_prot,pos_prot)

            #print(rep_prot)

            pred = model(rep_comp,rep_prot, rep_lig)
            z, pos  = [], [] 
            preds.append(pred.reshape((1,)))
            targets.append(torch.tensor(d['energy'], device=device).reshape((1,))) 
        loss = loss_func(torch.stack(preds), torch.stack(targets))
        loss.backward()
        optimizer.step()
        print('Training preds:', flush=True)
        print(preds, flush=True)
        print('targets: ', flush=True)
        print(targets, flush=True)
        preds, targets = [], []
        batch_losses.append(loss.item())

    return batch_losses

def val_loop(dataloader, model, loss_func, device):
    '''
    Validation loop
    '''
    with torch.no_grad():
        batch_losses = []
        for batch in dataloader:
            preds = []
            targets = []
            for d in batch:
                z = torch.tensor(d['z'], dtype=torch.long, device=device)
                pos = torch.tensor(d['pos'], device=device)
                lig_index=d['lig']
                prot_index = np.invert(d['lig'])
                z_lig = z[lig_index]
                pos_lig = pos[lig_index]
                z_prot = z[prot_index]
                pos_prot = pos[prot_index]

                rep_comp= (z,pos)
                rep_lig= (z_lig, pos_lig)
                rep_prot= (z_prot,pos_prot)
                pred = model(rep_comp,rep_prot, rep_lig)

                z, pos  = [], [] 
                preds.append(pred.reshape((1,)))
                targets.append(torch.tensor(d['energy'], device=device).reshape((1,))) 
            loss = loss_func(torch.stack(preds), torch.stack(targets))
            preds = []
            targets = []
            batch_losses.append(loss.item())
        return batch_losses


def train(model, train_dataloader, val_dataloader, n_epochs, loss_func, device,
    save_file=None, start_epoch=0, train_loss=[], test_loss=[]):
    '''
    Loop for training, validating, and saving model results
    '''
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch_idx in range(start_epoch, n_epochs):
        print('Epoch ' + str(epoch_idx + 1) + ' of ' + str(n_epochs), flush=True)
        print('training', flush=True)
        train_batch_losses = train_loop(train_dataloader, model, loss_func, optimizer, device)
        print(np.mean(train_batch_losses), flush=True)
        print('validating', flush=True)
        val_batch_losses= val_loop(val_dataloader, model, loss_func, device)
        print(np.mean(val_batch_losses), flush=True)

        train_loss.append(np.mean(train_batch_losses))
        test_loss.append(np.mean(val_batch_losses))

        if save_file is None:
            continue
        elif os.path.isdir(save_file):
            torch.save(model.state_dict(), f'{save_file}/{epoch_idx}.th')
            pkl.dump(np.vstack(train_loss),
                open(f'{save_file}/train_err.pkl', 'wb'))
            pkl.dump(np.vstack(test_loss),
                open(f'{save_file}/test_err.pkl', 'wb'))
        elif '{}' in save_file:
            torch.save(model.state_dict(), save_file.format(epoch_idx))
        else:
            torch.save(model.state_dict(), save_file)

    # print(model.state_dict())
    # print(model)

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
        **json.load(open(fn, 'r'))).compounds
    exp_compounds = [c for c in exp_compounds if c.achiral==achiral]

    affinity_dict = {c.compound_id: c.experimental_data['pIC50'] \
        for c in exp_compounds if 'pIC50' in c.experimental_data}

    return(affinity_dict)

def merge_labels_and_data(affinities, ds):
    '''
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
    '''
    final_ds = []
    for (_, compound_id), pose in ds:
        comp_dict = pose.copy()
        comp_dict['energy'] = affinities[compound_id]
        final_ds.append(comp_dict)
    
    return final_ds

def init_data(params):
    '''
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
    '''
    all_fns = glob(f'{params["dataset"]}/*complex.pdb')
    re_pat = r'(Mpro-P[0-9]{4}_0[AB]).*?([A-Z]{3}-[A-Z]{3}-.*?)_complex.pdb'
    compounds = [re.search(re_pat, fn).groups() for fn in all_fns]

    ## Load the experimental affinities
    exp_affinities = load_affinities(params["dataset_labels"])

    ## Trim docked structures and filenames to remove compounds that don't have
    ##  experimental data
    all_fns, compounds = zip(*[o for o in zip(all_fns, compounds) \
        if o[1][1] in exp_affinities])

    ds = DockedDataset(all_fns, compounds)

    total_ds = merge_labels_and_data(exp_affinities, ds)

    n_train = int(len(total_ds) * params["train_split"])
    n_val = int(len(total_ds) * params["val_split"])
    n_test = len(total_ds) - n_train - n_val

    print('Training size: ' + str(n_train), flush=True)
    print('Validation size: ' + str(n_val), flush=True)
    print('Testing size: ' + str(n_test), flush=True)


    batch_size = params["batch_size"]

    train_data, val_data, test_data = random_split(
    total_ds, [n_train, n_val, n_test], torch.Generator().manual_seed(params["random_seed"])
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



    if params["qm9"] is not None:
        print('Using QM9')
        qm9_path = params['qm9']
        qm9_dataset = QM9(qm9_path)
        model_qm9, _ = PyGSchNet.from_qm9_pretrained(qm9_path, qm9_dataset, target=10)

        wts = model_qm9.state_dict()

        ## Get rid of entries in state_dict that correspond to atomref
        wts = {k: v for k,v in model_qm9.state_dict().items() \
            if 'atomref' not in k}

        m = SchNet()
        m.load_state_dict(wts)
        m.cutoff = 3.5
        model = SchNet.get_model(model=m, strategy='delta eV')

    else:
        if params["model"] == 'schnet':
            m = SchNet()
            if params["model_in"] is not None:
                wts_fn = find_most_recent(params["model_in"])[1]
                m.load_state_dict(torch.load(wts_fn))
            m.cutoff = 3.5
            model = SchNet.get_model(model=m, strategy='delta hartree')

    ## Load model weights as necessary
    if params["cont"]:
        print("continuing")
        start_epoch, wts_fn = find_most_recent(params["model_o"])
        #print(start_epoch)
        model.load_state_dict(torch.load(wts_fn))

        ## Load error dicts
        if os.path.isfile(f'{params["model_o"]}/train_err.pkl'):
            train_loss = pkl.load(open(f'{params["model_o"]}/train_err.pkl',
                'rb')).tolist()
        else:
            train_loss = []
        if os.path.isfile(f'{params["model_o"]}/test_err.pkl'):
            test_loss = pkl.load(open(f'{params["model_o"]}/test_err.pkl',
                'rb')).tolist()
        else:
            test_loss = []

        ## Need to add 1 to start_epoch bc the found idx is the last epoch
        ##  successfully trained, not the one we want to start at
        start_epoch += 1
    else:
        start_epoch = 0
        train_loss = []
        test_loss = []

    print('Beginning training', flush=True)
    
    model, train_loss, test_loss = train(model, train_dataloader, val_dataloader, params["n_epochs"], 
        loss_func, device, params["model_o"], start_epoch, train_loss, test_loss)
    
    if params["plot_o"] is not None:
        plot_loss(train_loss.mean(axis=1), test_loss.mean(axis=1), params["plot_o"])

if __name__ == "__main__":
    main()


