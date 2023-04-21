import sys

sys.path.append('../')
import torch
import torch.nn as nn
from pathos.multiprocessing import ProcessPool
import numpy as np
import os
from tqdm import tqdm
import dill as pickle  # 使用 dill 库代替标准库中的 pickle 库
import rdkit
from rdkit import Chem
import math, random, sys
from optparse import OptionParser
import pickle

from fast_jtnn.mol_tree import MolTree  # 导入 MolTree


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

def convert(test_data_path, pool):
    with open(test_data_path) as f:
        lines = f.readlines()
        data = [line.strip().split(',')[0] for line in lines[1:]]  # 跳过第一行，获取 SMILES 列的数据
    print('Input File read')

    print('Tensorizing .....')
    preprocess_data = pool.map(tensorize, data)

    pool.close()
    pool.join()
    print('Tensorizing Complete')

    return preprocess_data


if __name__ == "__main__":
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    parser = OptionParser()
    parser.add_option("-t", "--train", dest="test_data_path", default="/home/dell/wangzhen/RealQED(2.17)/test/data/finetune/freesolv/freesolv.csv")
    parser.add_option("-j", "--jobs", dest="njobs", default=40, help="48")
    parser.add_option("-o", "--output", dest="output_path",
                      default="/home/dell/wangzhen/RealQED(2.17)/test/data/finetune/freesolv_processed")

    opts, args = parser.parse_args()
    opts.njobs = int(opts.njobs)

    pool = ProcessPool(opts.njobs)  # 使用 ProcessPool
    preprocess_data = convert(opts.test_data_path, pool)
    # 将处理后的数据保存到文件
    with open(opts.output_path, 'wb') as f:
        pickle.dump(preprocess_data, f)

    print(f"Processed data saved to: {opts.output_path}")
