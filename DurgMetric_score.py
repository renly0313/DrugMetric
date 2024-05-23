import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import math, random
import numpy as np
import argparse
import pickle as pickle
from fast_jtnn import *
import rdkit
from rdkit import Chem
from tqdm import tqdm
import os
import pandas as pd

from autogluon.tabular import TabularDataset, TabularPredictor
import numpy as np


def tensorize(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
    except:
        print('Failed to convert SMILES to molecule:', smiles)
        return None
    if mol is None:
        print('Invalid SMILES:', smiles)
        return None

    mol_tree = MolTree(smiles)
    mol_tree.recover()
    mol_tree.assemble()
    for node in mol_tree.nodes:
        if node.label not in node.cands:
            node.cands.append(node.label)

    del mol_tree.mol
    for node in mol_tree.nodes:
        del node.mol

    return mol_tree


def main_vae_inference(train_path, vocab_path, model_path, batch_size=8, latent_size=64):
    vocab = [x.strip("\r\n ") for x in open(vocab_path)]
    vocab = Vocab(vocab)

    model = JTNNVAE(vocab, hidden_size=450, latent_size=latent_size, depthT=20, depthG=3).cuda()
    model.load_state_dict(torch.load(model_path))
    print("Model #Params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,))

    mol_tree_mean = []
    train_processed_list = []
    smiles_success = []

    with torch.no_grad():
        data = pd.read_csv(train_path, header=None)
        data = data.iloc[:, 0].values.tolist()

        for smiles in data:
            try:
                mol_tree = tensorize(smiles)
                if mol_tree is None:
                    continue
                mol_tree = [mol_tree]
                loader = MolTreeFolder(mol_tree, vocab, batch_size)
                for batch in loader:
                    try:
                        mol_tree_mean = model.forward_dis(batch)
                        mol_tree_mean = mol_tree_mean.cpu().detach().numpy()
                        train_processed_list.append(mol_tree_mean)
                        smiles_success.append(smiles)
                    except Exception as e:
                        print("Error while processing molecule:", smiles)
                        print(e)
                        continue
            except Exception as e:
                print("Error while converting SMILES to MolTree:", smiles)
                print(e)
                continue

        if len(train_processed_list) == 0:
            print("No valid molecules found in the input data.")

    train_processed_list = np.vstack(train_processed_list)

    return train_processed_list, smiles_success


if __name__ == '__main__':
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str,
                        default="molecule_smiles_files.csv",
                        help='Input file path')
    parser.add_argument('--vocab', type=str, default="/data/vocab/all_data_vocab.txt",
                        help='Vocab file path')
    parser.add_argument('--model_path', type=str,
                        default="fast_molvae/vae_model/model.epoch-19",
                        help='Model file path')
    parser.add_argument('--latent_size', type=int, default=64, help='Latent size for VAE')

    args = parser.parse_args()
    latents, smiles_list = main_vae_inference(args.train, args.vocab, args.model_path, args.latent_size)

    # Assuming smiles_list and latents are defined
    output_df = pd.DataFrame({
        'SMILES': smiles_list,
        'Latent': [list(latent) for latent in latents]
    })

    # Convert Latent column's list to individual columns in a DataFrame and set column names
    num_features = len(output_df['Latent'].iloc[0])  # Assuming all lists in 'Latent' are of the same length
    column_names = [str(i) for i in range(num_features)]  # Create column names as strings from '0' to 'num_features-1'
    latent_df = pd.DataFrame(output_df['Latent'].tolist(), columns=column_names)

    # Load the previously trained model
    predictor = TabularPredictor.load(
        '/autogluen/AutogluonModels/ag-20230308_124517/',
        require_py_version_match=False
    )

    # Use the transformed DataFrame as TabularDataset
    test_data = TabularDataset(latent_df)

    # Predict, clip results between 0 and 100, and round to two decimals
    y_pred = np.around(np.clip(predictor.predict(test_data).values, 0, 100), 2)

    # Convert predictions to DataFrame
    y_pred = pd.DataFrame(y_pred, columns=['DrugMetric'])

    # Save the results to a CSV file
    y_pred.to_csv(
        'your/path/to/save',
        index=False
    )

