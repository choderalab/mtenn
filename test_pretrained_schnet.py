import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from mtenn.conversion_utils.schnet import SchNet
from mtenn.conversion_utils.e3nn import E3NN
import yaml
import ast
import argparse
from covid_moonshot_ml.utils import find_most_recent, plot_loss
from e3nn import o3
from torch_cluster import radius_graph
from collections import Counter
from covid_moonshot_ml.data.dataset import DockedDataset
from glob import glob
import re
import json
from covid_moonshot_ml.schema import ExperimentalCompoundDataUpdate
torch.manual_seed(42)

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
        "dataset_labels" : config["dataset_labels"],
        "train_split" : config["train_split"],
        "val_split" : config["val_split"],
        "random_seed" : config["random_seed"],
        "batch_size" : config["batch_size"],
        "schnet spice" : config["schnet spice"],
        "schnet no pretrain" : config["schnet no pretrain"],
        "schnet qm9" : config["schnet qm9"],
        
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


def eval_loop(dataloader, model, loss_func, device):
    '''
    Evaluate model on test set
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

                #calculates MAE but can use a loss_func instead
                batch_losses.append(np.absolute(pred.item() - d['energy']))

        return batch_losses


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


    # if params["model"] == "e3nn":
    #     spl = add_one_hot_encodings(spl)

    n_train = int(len(total_ds) * params["train_split"])
    n_val = int(len(total_ds) * params["val_split"])
    n_test = len(total_ds) - n_train - n_val

    batch_size = params["batch_size"]

    train_data, val_data, test_data = random_split(
    total_ds, [n_train, n_val, n_test], torch.Generator().manual_seed(params["random_seed"])
    )

    def collate_fn(batch):
        return batch

    train_dataloader = DataLoader(
        train_data, batch_size=n_train, shuffle=False, collate_fn=collate_fn
    )
    val_dataloader = DataLoader(
        val_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    test_dataloader = DataLoader(
        test_data, batch_size=n_test, shuffle=False, collate_fn=collate_fn
    )


    return train_dataloader, val_dataloader, test_dataloader


def main():
    args = get_args()
    params = get_params(args.config)

    loss_func = torch.nn.L1Loss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_dataloader, val_dataloader, test_dataloader = init_data(params)


    m = SchNet()
    model = SchNet.get_model(model=m, strategy='delta hartree')
    model.load_state_dict(torch.load("/lila/home/tyckoa/research/results/pretrained schnet spice/650.th"))
    model.representation.cutoff = 3.5
    print(model)
    print(model.state_dict())
    model.eval()

    model.to(device)

    #Prints MAE values for  each modelsx

    output = eval_loop(test_dataloader, model=model, loss_func=loss_func, device=device)
    print('SPICE:')
    print(np.mean(output))


    m = SchNet()
    model = SchNet.get_model(model=m, strategy='delta eV')
    model.load_state_dict(torch.load("/lila/home/tyckoa/research/results/pretrained schnet qm9/715.th"))
    model.representation.cutoff = 3.5
    model.eval()

    model.to(device)

    output = eval_loop(test_dataloader, model=model, loss_func=loss_func, device=device)
    print('QM9:')
    print(np.mean(output))

    m = SchNet()
    model = SchNet.get_model(model=m, strategy='delta eV')
    model.load_state_dict(torch.load("/lila/home/tyckoa/research/results/schnet no pretraining/857.th"))
    model.representation.cutoff = 3.5
    model.eval()

    model.to(device)

    output = eval_loop(test_dataloader, model=model, loss_func=loss_func, device=device)
    print("No pretrain:")
    print(np.mean(output))

 
if __name__ == "__main__":
    main()


