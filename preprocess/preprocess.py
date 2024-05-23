import sys
import os
import pickle
from multiprocessing import Pool
from optparse import OptionParser
from tqdm import tqdm
import numpy as np
import rdkit
from utils import tensorize  

def convert(train_path, pool, num_splits, output_path):
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    out_path = os.path.join(output_path, './')
    if os.path.isdir(out_path) is False:
        os.makedirs(out_path)

    with open(train_path) as f:
        data = [line.strip("\r\n ").split()[0] for line in f]
    print('Input File read')

    print('Tensorizing .....')
    all_data = pool.map(tensorize, data)
    pool.close()
    pool.join()
    all_data_split = np.array_split(all_data, num_splits)
    print('Tensorizing Complete')

    for split_id in tqdm(range(num_splits)):
        with open(os.path.join(output_path, 'tensors-%d.pkl' % split_id), 'wb') as f:
            pickle.dump(all_data_split[split_id], f)

    return True


def main_preprocess(train_path, output_path, num_splits=10, njobs=os.cpu_count()):
    pool = Pool(njobs)
    convert(train_path, pool, num_splits, output_path)
    return True


if __name__ == "__main__":
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    parser = OptionParser()
    parser.add_option("-t", "--train", dest="train_path", default="/test/data/admet-score/bbb.csv")
    parser.add_option("-n", "--split", dest="nsplits", default=1, help="10")
    parser.add_option("-j", "--jobs", dest="njobs",default=40, help="8")
    parser.add_option("-o", "--output", dest="output_path",default="/test/data/admet-score/bbb_processed")
    opts, args = parser.parse_args()  

    opts.njobs = int(opts.njobs)  
    pool = Pool(opts.njobs)  

    num_splits = int(opts.nsplits)  
    convert(opts.train_path, pool, num_splits, opts.output_path)  
