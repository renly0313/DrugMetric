import sys
import os
import pickle
from multiprocessing import Pool
from optparse import OptionParser
from tqdm import tqdm
import numpy as np
import rdkit
from utils import tensorize  # 从utils.py文件导入tensorize函数

# convert函数用于将数据集中的每个SMILES字符串转换成MolTree对象，并将结果保存为pickle文件
def convert(train_path, pool, num_splits, output_path):
    # 配置RDKit日志记录器，避免输出烦人的警告
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    # 创建输出目录
    out_path = os.path.join(output_path, './')
    if os.path.isdir(out_path) is False:
        os.makedirs(out_path)

    # 读取训练集文件中的SMILES字符串
    with open(train_path) as f:
        data = [line.strip("\r\n ").split()[0] for line in f]
    print('Input File read')

    # 将SMILES字符串转换为MolTree对象
    print('Tensorizing .....')
    all_data = pool.map(tensorize, data)
    pool.close()
    pool.join()
    all_data_split = np.array_split(all_data, num_splits)
    print('Tensorizing Complete')

    # 将MolTree对象保存为pickle文件
    for split_id in tqdm(range(num_splits)):
        with open(os.path.join(output_path, 'tensors-%d.pkl' % split_id), 'wb') as f:
            pickle.dump(all_data_split[split_id], f)

    return True


# main_preprocess函数是程序的主要逻辑，用于调用convert函数进行数据预处理
def main_preprocess(train_path, output_path, num_splits=10, njobs=os.cpu_count()):
    pool = Pool(njobs)
    convert(train_path, pool, num_splits, output_path)
    return True


if __name__ == "__main__":
    # 配置RDKit日志记录器，避免输出烦人的警告
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    # 使用OptionParser解析命令行参数
    parser = OptionParser()
    parser.add_option("-t", "--train", dest="train_path", default="/home/dell/wangzhen/RealQED(2.17)/test/data/docking/affinity_large_than_8.csv")
    parser.add_option("-n", "--split", dest="nsplits", default=10, help="10")
    parser.add_option("-j", "--jobs", dest="njobs",default=40, help="8")
    parser.add_option("-o", "--output", dest="output_path",default="/home/dell/wangzhen/RealQED(2.17)/test/data/docking/affinity_large_than_8_processed")
    opts, args = parser.parse_args()  # 使用OptionParser进行命令行参数解析

    opts.njobs = int(opts.njobs)  # 将njobs参数从字符串类型转换成整数类型
    pool = Pool(opts.njobs)  # 使用multiprocessing库创建进程池，以并行化处理数据

    num_splits = int(opts.nsplits)  # 将nsplits参数从字符串类型转换成整数类型
    convert(opts.train_path, pool, num_splits, opts.output_path)  # 将命令行参数传入convert函数进行数据处理
