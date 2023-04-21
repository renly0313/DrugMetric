import rdkit
import sys
sys.path.append('../../')
import time
import argparse
from argparse import Namespace
from logging import Logger
import torch
from fast_jtnn.datautils import get_data_processed
if __name__ == '__main__':
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    get_data_processed(args.data_path)
    # parser.add_argument('--train', required=True)
    # parser.add_argument('--vocab', required=True)
    # parser.add_argument('--save_dir', required=True)
    parser.add_argument('--data_path', type=str, default="../data/gdb_save_data/2gdb_total_mean_std.pkl")

