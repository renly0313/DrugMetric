import torch
import torch.nn as nn
from torch.autograd import Variable

import math, random, sys
from optparse import OptionParser
from collections import deque

import rdkit
import rdkit.Chem as Chem

from fast_jtnn import *

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = OptionParser()
parser.add_option("-t", "--test", dest="test_path", default="../data/datasets/new_data/clinical_drug/clinical_drug_train.txt")
parser.add_option("-v", "--vocab", dest="vocab_path", default="../data/vocab/all_data_vocab.txt")
parser.add_option("-m", "--model", dest="model_path", default="../data/save_model/pre_zinc250_mix_rand1_processed_model1206/model.epoch-19")
parser.add_option("-w", "--hidden", dest="hidden_size", default=450)
parser.add_option("-l", "--latent", dest="latent_size", default=64)
parser.add_option("-T", "--depthT", dest="depthT", default=20)
parser.add_option("-G", "--depthG", dest="depthG", default=3)
opts, args = parser.parse_args()

vocab = [x.strip("\r\n ") for x in open(opts.vocab_path)]
vocab = Vocab(vocab)

hidden_size = int(opts.hidden_size)
latent_size = int(opts.latent_size)
depthT = int(opts.depthT)
depthG = int(opts.depthG)
#stereo = True if int(opts.stereo) == 1 else False

model = JTNNVAE(vocab, hidden_size, latent_size, depthT, depthG)
model.load_state_dict(torch.load(opts.model_path))
model = model.cuda()

data = []
with open(opts.test_path) as f:
    for line in f:
        s = line.strip("\r\n ").split()[0]
        data.append(s)

acc = 0.0
tot = 0
for smiles in data:
    mol = Chem.MolFromSmiles(smiles)
    smiles3D = Chem.MolToSmiles(mol, isomericSmiles=False)
    dec_smiles = model.reconstruct(smiles3D)
    recon_smiles = []
    if dec_smiles == smiles3D:
        acc += 1
    tot += 1
    recon_smiles.append(dec_smiles)
    print(acc / tot, dec_smiles)
recon_smiles.to_csv("../data/reconstruct_smiles/pre_zinc250_mix_rand1_processed_model1206_l64.txt", index=None)
"""
    dec_smiles = model.recon_eval(smiles3D)
    tot += len(dec_smiles)
    for s in dec_smiles:
        if s == smiles3D:
            acc += 1
    print acc / tot
"""