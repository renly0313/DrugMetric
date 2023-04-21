import torch
from torch.utils.data import Dataset, DataLoader
from fast_jtnn.mol_tree import MolTree
import numpy as np
from fast_jtnn.jtnn_enc import JTNNEncoder
from fast_jtnn.mpn import MPN
from fast_jtnn.jtmpn import JTMPN
import pickle as pickle
import os, random
import pickle
class PairTreeFolder(object):

    def __init__(self, data_folder, vocab, batch_size, num_workers=4, shuffle=True, y_assm=True, replicate=None):
        self.data_folder = data_folder
        self.data_files = [fn for fn in os.listdir(data_folder)]
        self.batch_size = batch_size
        self.vocab = vocab
        self.num_workers = num_workers
        self.y_assm = y_assm
        self.shuffle = shuffle

        if replicate is not None: #expand is int
            self.data_files = self.data_files * replicate

    def __iter__(self):
        for fn in self.data_files:
            fn = os.path.join(self.data_folder, fn)
            with open(fn, 'rb') as f:
                data = pickle.load(f)

            if self.shuffle: 
                random.shuffle(data) #shuffle data before batch 在批处理前对数据进行洗牌

            batches = [data[i : i + self.batch_size] for i in range(0, len(data), self.batch_size)]
            if len(batches[-1]) < self.batch_size:
                batches.pop()

            dataset = PairTreeDataset(batches, self.vocab, self.y_assm)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x:x[0])#, num_workers=self.num_workers)

            for b in dataloader:
                yield b

            del data, batches, dataset, dataloader

#  修改后的版本
class MolTreeFolder(object):

    def __init__(self, data, vocab, batch_size, num_workers=4, shuffle=False, assm=True):
        if isinstance(data, str): # data is a file path
            self.data_files = [fn for fn in os.listdir(data)]
            self.data = None
        elif isinstance(data, list): # data is a list
            self.data_files = None
            self.data = data
        else:
            raise ValueError("Data must be either a file path or a list")

        self.batch_size = batch_size
        self.vocab = vocab
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.assm = assm

    def __iter__(self):
        if self.data_files is not None:
            for fn in self.data_files:
                fn = os.path.join(self.data_folder, fn)
                with open(fn, 'rb') as f:
                    data = pickle.load(f)
        else:
            data = self.data

        if self.shuffle:
            random.shuffle(data) #shuffle data before batch

        batches = [data[i: i + self.batch_size] for i in range(0, len(data), self.batch_size)]

        dataset = MolTreeDataset(batches, self.vocab, self.assm)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0])

        for b in dataloader:
            yield b

        #free up memory
        del data, batches, dataset, dataloader

# class MolTreeFolder(object):
#
#     def __init__(self, data_folder, vocab, batch_size, num_workers=4, shuffle=False, assm=True, replicate=None):
#         self.data_folder = data_folder
#         self.data_files = [fn for fn in os.listdir(data_folder)]
#         self.batch_size = batch_size
#         self.vocab = vocab
#         self.num_workers = num_workers
#         self.shuffle = shuffle
#         self.assm = assm
#
#         if replicate is not None: #expand is int
#             self.data_files = self.data_files * replicate
#
#     def __iter__(self):
#         for fn in self.data_files:
#             fn = os.path.join(self.data_folder, fn)
#             with open(fn, 'rb') as f:
#                 data = pickle.load(f)
#
#             if self.shuffle:
#                 random.shuffle(data) #shuffle data before batch
#
#             batches = [data[i: i + self.batch_size] for i in range(0, len(data), self.batch_size)]
#             # if len(batches[-1]) < self.batch_size:
#             #     batches.pop()
#
#             dataset = MolTreeDataset(batches, self.vocab, self.assm)
#             dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0])#, num_workers=self.num_workers)
#
#             for b in dataloader:
#                 yield b
#
#             #free up memory 释放内存
#             del data, batches, dataset, dataloader






class PairTreeDataset(Dataset):

    def __init__(self, data, vocab, y_assm):
        self.data = data
        self.vocab = vocab
        self.y_assm = y_assm

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        batch0, batch1 = list(zip(*self.data[idx]))
        return tensorize(batch0, self.vocab, assm=False), tensorize(batch1, self.vocab, assm=self.y_assm)

class MolTreeDataset(Dataset):

    def __init__(self, data, vocab, assm=True):
        self.data = data
        self.vocab = vocab
        self.assm = assm

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return tensorize(self.data[idx], self.vocab, assm=self.assm)

def tensorize(tree_batch, vocab, assm=True):
    # 为当前batch_size的每个mol_tree的每个node添加idx和wid属性。
    # node.idx为当前mol_tree的node在整个batch_size的所有tree的所有node组成的list中的idx(0,1,2,3...)；
    # node.wid为当前node在vocab中对应的idx
    set_batch_nodeID(tree_batch, vocab)

    smiles_batch = [tree.smiles for tree in tree_batch]
    jtenc_holder, mess_dict = JTNNEncoder.tensorize(tree_batch)
    jtenc_holder = jtenc_holder
    mpn_holder = MPN.tensorize(smiles_batch)#mpn_holder = (fatoms, fbonds, agraph, bgraph, scope)

    if assm is False:
        return tree_batch, jtenc_holder, mpn_holder

    cands = []
    batch_idx = []
    for i, mol_tree in enumerate(tree_batch):
        for node in mol_tree.nodes:
            #Leaf node's attachment is determined by neighboring node's attachment
            # 叶子节点的连接节点是由邻居节点的连接节点决定的
            if node.is_leaf or len(node.cands) == 1:
                continue
            cands.extend([(cand, mol_tree.nodes, node) for cand in node.cands])
            batch_idx.extend([i] * len(node.cands))

    jtmpn_holder = JTMPN.tensorize(cands, mess_dict)#jtmpn_holder = (fatoms, fbonds, agraph, bgraph, scope)
    batch_idx = torch.LongTensor(batch_idx)

    return tree_batch, jtenc_holder, mpn_holder, (jtmpn_holder, batch_idx)

def set_batch_nodeID(mol_batch, vocab):
    tot = 0
    for mol_tree in mol_batch:
        for node in mol_tree.nodes:
            node.idx = tot
            node.wid = vocab.get_index(node.smiles)#Get index of the corresponding cluster vocabulary item. 获取相应的集群词汇项的索引。
            tot += 1

def get_data_processed(data_path, args):
    print('read processed data')
    data_path = os.path.join(data_path, 'mol_tree_mean={}_mol_tree_std={}'.format(args.mol_tree_mean,  args.mol_tree_std))
    with open(os.path.join(data_path, '2gdb_total_mean_std.pkl'),'rb') as f:
        train_data = pickle.load(f)
    print('Read processed data finish')
    return train_data

def random_sample_from_txt_file(txt_file_path, random_seed):
    with open(txt_file_path, 'r') as f:
        lines = f.readlines()
    random.seed(random_seed)
    sampled_lines = random.sample(lines, 4527)
    sampled_molecules = [line.strip() for line in sampled_lines]

    # 从文件路径中获取文件名和文件夹路径
    file_name = os.path.basename(txt_file_path)
    folder_path = os.path.dirname(txt_file_path)

    # 创建保存结果的文件夹
    result_folder_path = os.path.join(folder_path, 'origin_sample')
    os.makedirs(result_folder_path, exist_ok=True)

    # 将采样的分子写入新的文件中
    result_file_path = os.path.join(result_folder_path, f"{file_name}_{random_seed}")
    with open(result_file_path, 'w') as f:
        for molecule in sampled_molecules:
            f.write(molecule + '\n')

    return sampled_molecules
