#将tankbind源文件夹路径添加到系统路径
tankbind_src_folder_path ="../tankbind/"
import sys
sys.path.insert(0, tankbind_src_folder_path)

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
base_pre = f"/home/dell/wangzhen/TankBind-main/examples/HTVS/"

from rdkit import Chem
from rdkit.Chem import PandasTools

# 导入torch库并设置线程数为1
import torch
torch.set_num_threads(1)

from torch_geometric.data import Dataset
from utils import construct_data_from_graph_gvp
from feature_utils import extract_torchdrug_feature_from_mol, get_canonical_smiles

info = pd.read_csv('/home/dell/wangzhen/TankBind-main/examples/HTVS/info/info.csv')
# 创建一个用于虚拟筛选的自定义数据集类，继承自torch_geometric.data中的Dataset
class MyDataset_VS(Dataset):

    # 初始化函数，设置数据集的相关参数
    def __init__(self, root, data=None, protein_dict=None, proteinMode=0, compoundMode=1,
                 pocket_radius=20, shake_nodes=None,
                 transform=None, pre_transform=None, pre_filter=None):

        # 存储输入数据
        self.data = data

        # 存储蛋白质特征字典
        self.protein_dict = protein_dict

        # 调用父类构造函数
        super().__init__(root, transform, pre_transform, pre_filter)

        # 输出处理后的文件路径
        print(self.processed_paths)

        # 加载处理后的数据
        self.data = torch.load(self.processed_paths[0])

        # 加载处理后的蛋白质特征字典
        self.protein_dict = torch.load(self.processed_paths[1])

        # 设置蛋白质模式
        self.proteinMode = proteinMode

        # 设置口袋半径
        self.pocket_radius = pocket_radius

        # 设置化合物模式
        self.compoundMode = compoundMode

        # 设置抖动节点的参数
        self.shake_nodes = shake_nodes

    @property
    # 定义处理后的文件名属性
    def processed_file_names(self):

        # 返回处理后的文件名列表
        return ['data.pt', 'protein.pt']

    # 处理数据的函数
    def process(self):

        # 保存处理后的数据
        torch.save(self.data, self.processed_paths[0])

        # 保存处理后的蛋白质特征字典

        torch.save(self.protein_dict, self.processed_paths[1])

    # 定义数据集长度函数
    def len(self):

        # 返回数据集长度
        return len(self.data)

    # 定义获取数据的函数
    def get(self, idx):
        # 获取指定索引的数据行
        line = self.data.iloc[idx]

        # 提取smiles信息
        smiles = line['smiles']

        # 提取口袋中心坐标
        pocket_com = line['pocket_com']
        pocket_com = np.array(pocket_com.split(",")).astype(float) if type(pocket_com) == str else pocket_com
        pocket_com = pocket_com.reshape((1, 3))

        # 判断是否使用整个蛋白质
        use_whole_protein = line['use_whole_protein'] if "use_whole_protein" in line.index else False
        # 提取蛋白质名称
        protein_name = line['protein_name']

        # 从蛋白质字典中获取蛋白质相关数据
        protein_node_xyz, protein_seq, protein_node_s, protein_node_v, protein_edge_index, protein_edge_s, protein_edge_v = \
        self.protein_dict[protein_name]

        # 尝试处理smiles信息
        try:
            smiles = get_canonical_smiles(smiles)
            mol = Chem.MolFromSmiles(smiles)
            mol.Compute2DCoords()
            coords, compound_node_features, input_atom_edge_list, input_atom_edge_attr_list, pair_dis_distribution = extract_torchdrug_feature_from_mol(
                mol, has_LAS_mask=True)
        except:
            # 如果处理过程中出现错误，使用占位符smiles 'CCC'替代，并输出错误信息
            print("something wrong with ", smiles,
                  "to prevent this stops our screening, we repalce it with a placeholder smiles 'CCC'")
            smiles = 'CCC'
            mol = Chem.MolFromSmiles(smiles)
            mol.Compute2DCoords()
            coords, compound_node_features, input_atom_edge_list, input_atom_edge_attr_list, pair_dis_distribution = extract_torchdrug_feature_from_mol(
                mol, has_LAS_mask=True)
        # y is distance map, instead of contact map.
        # 使用从图中构建的数据创建data对象
        data, input_node_list, keepNode = construct_data_from_graph_gvp(protein_node_xyz, protein_seq, protein_node_s,
                                                                        protein_node_v, protein_edge_index,
                                                                        protein_edge_s, protein_edge_v,
                                                                        coords, compound_node_features,
                                                                        input_atom_edge_list, input_atom_edge_attr_list,
                                                                        pocket_radius=self.pocket_radius,
                                                                        use_whole_protein=use_whole_protein,
                                                                        includeDisMap=True,
                                                                        use_compound_com_as_pocket=False,
                                                                        chosen_pocket_com=pocket_com,
                                                                        compoundMode=self.compoundMode)
        # 更新data对象的compound_pair属性
        data.compound_pair = pair_dis_distribution.reshape(-1, 16)

        return data


from feature_utils import get_protein_feature
from Bio.PDB import PDBParser
from feature_utils import get_clean_res_list
#创建PDB解析器实例
parser = PDBParser(QUIET=True)

#创建空字典以存储蛋白质特征
protein_dict = {}

#设置蛋白质名
proteinName = "7kjs"

#设置蛋白质文件路径
proteinFile = f"{base_pre}/{proteinName}.pdb"

#解析PDB文件获取蛋白质结构
s = parser.get_structure("example", proteinFile)

#获取蛋白质残基列表
res_list = list(s.get_residues())

#清洗残基列表，确保每个残基中都存在α碳
clean_res_list = get_clean_res_list(res_list, ensure_ca_exist=True)

#计算清洗后残基列表的蛋白质特征，并将结果存储到protein_dict字典中，以蛋白质名作为键
protein_dict[proteinName] = get_protein_feature(clean_res_list)

dataset_path = f"{base_pre}/dataset/"
os.system(f"rm -r {dataset_path}")
os.system(f"mkdir -p {dataset_path}")
dataset = MyDataset_VS(dataset_path, data=info, protein_dict=protein_dict)


import logging
from torch_geometric.loader import DataLoader
from tqdm import tqdm    # pip install tqdm if fails.
from model import get_model

batch_size = 8
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device= 'cpu'
logging.basicConfig(level=logging.INFO)
model = get_model(0, logging, device)
model = model.to(device)  # 将模型转移到 GPU
# modelFile = "../saved_models/re_dock.pt"
# self-dock model
modelFile = "../saved_models/self_dock.pt"

model.load_state_dict(torch.load(modelFile, map_location=device))
_ = model.eval()

# 创建数据加载器
data_loader = DataLoader(dataset, batch_size=batch_size, follow_batch=['x', 'y', 'compound_pair'], shuffle=False, num_workers=8)

# 初始化亲和力预测列表和距离预测列表
affinity_pred_list = []
y_pred_list = []

import torch
import gc
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

# 初始化 NVIDIA 管理库
nvmlInit()

# 获取当前 GPU 设备句柄
device_handle = nvmlDeviceGetHandleByIndex(torch.cuda.current_device())

def get_free_memory():
    # 获取当前 GPU 的内存信息
    mem_info = nvmlDeviceGetMemoryInfo(device_handle)
    return mem_info.free / (1024 * 1024)  # 返回以 MiB 为单位的可用内存

# 对数据加载器中的每个数据进行预测
for data in tqdm(data_loader):
    try:
        data = data.to(device)
        y_pred, affinity_pred = model(data)
        affinity_pred_list.append(affinity_pred.detach().cpu())

        # # 对于高通量虚拟筛选，我们不需要保存预测的距离图
        if False:
            # we don't need to save the predicted distance map in HTVS setting.
            for i in range(data.y_batch.max() + 1):
                y_pred_list.append((y_pred[data['y_batch'] == i]).detach().cpu())
        # 当可用显存小于 20000 MiB 时清除缓存
        if get_free_memory() < 20000:
            torch.cuda.empty_cache()
            gc.collect()  # 强制垃圾回收以进一步释放内存
    except Exception as e:
        print(e)
        continue
# 将预测结果整合到一个张量中
affinity_pred_list = torch.cat(affinity_pred_list)

# 将预测的亲和力值添加到info中
info = dataset.data
info['affinity'] = affinity_pred_list
#保存预测结果到CSV文件
info.to_csv('/home/dell/wangzhen/TankBind-main/examples/HTVS/info/info.csv')
