import sys

sys.path.append('../../../')
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import math, random
import numpy as np
from pathos.multiprocessing import ProcessPool
import argparse
import pickle as pickle
from fast_jtnn import *
import rdkit
from tqdm import tqdm
import os
from preprocess.preprocess_to_train.code.preprocess_jianhua import convert

def main_vae_inference(test_data_path,
                       njobs,
                       vocab,
                       model_path,
                       output_file,
                       batch_size=128):
    vocab = [x.strip("\r\n ") for x in open(vocab)]
    vocab = Vocab(vocab)

    model = JTNNVAE(vocab, hidden_size=450, latent_size=64, depthT=20, depthG=3).cuda()
    model.load_state_dict(torch.load(model_path))

    mol_tree_mean = []
    train_processed_list = []
    pool = ProcessPool(njobs)
    preprocess_data = convert(test_data_path, pool)
    with torch.no_grad():
        loader = MolTreeFolder(preprocess_data, vocab, batch_size)
        for batch in loader:
            try:
                mol_tree_mean = model.forward_dis(batch)
                mol_tree_mean = mol_tree_mean.cpu().detach().numpy()
                train_processed_list.append(mol_tree_mean)
            except Exception as e:
                print(e)
                continue

    train_processed_list = np.vstack(train_processed_list)

    output_path = os.path.join(output_file, args.output_file)
    with open(output_path, 'wb') as f:
        pickle.dump(train_processed_list, f)

    return train_processed_list


if __name__ == '__main__':
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    parser = argparse.ArgumentParser()

    parser.add_argument('--test_data_path', type=str,
                        default="/home/dell/wangzhen/RealQED(2.17)/test/data/PK/pk_data.csv")
    parser.add_argument("--njobs", default=40, help="48")
    parser.add_argument('--vocab', type=str, default="/home/dell/wangzhen/RealQED(2.17)/data/vocab/all_data_vocab.txt")
    parser.add_argument('--model_path', type=str,
                        default="/home/dell/wangzhen/RealQED(2.17)/data/save_model/pre_zinc250_mix_rand1_processed_model1206/pre_zinc250_mix_rand1_processed_model1206best_model.pkl")
    parser.add_argument('--output_file', type=str, default="/home/dell/wangzhen/RealQED(2.17)/test/data/PK/pk_data.pkl", help='Output file path')


    args = parser.parse_args()
    main_vae_inference(args.test_data_path,
                       args.njobs,
                       args.vocab,
                       args.model_path,
                       args.output_file)

