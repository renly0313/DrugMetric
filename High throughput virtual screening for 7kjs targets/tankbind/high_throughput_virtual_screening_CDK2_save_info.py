#将tankbind源文件夹路径添加到系统路径
tankbind_src_folder_path ="../tankbind/"
import sys
sys.path.insert(0, tankbind_src_folder_path)

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
base_pre = f"/home/dell/wangzhen/TankBind-main/examples/HTVS/"

import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools


# 读取 SDF 文件
sdf_file = "/home/dell/wangzhen/TankBind-main/examples/HTVS/Enamine_hts_collection_202303.sdf"
supplier = Chem.SDMolSupplier(sdf_file)

# 将分子添加到 DataFrame 中
mols = []
for mol in supplier:
    if mol is not None:
        mols.append(mol)

# 创建一个新的 DataFrame
m = PandasTools.LoadSDF(sdf_file, molColName="Molecule", includeFingerprints=False)

#添加分子 SMILES
m['SMILES'] = [Chem.MolToSmiles(mol) for mol in mols]

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

#创建一个存储蛋白质列表的文件
ds = f"{base_pre}/protein_list.ds"
with open(ds, "w") as out:
    out.write(f"/{proteinName}.pdb\n")

p2rank = "/home/dell/wangzhen/TankBind-main/tankbind/p2rank_2.3/prank"

#设置p2rank工具的路径
cmd = f"{p2rank} predict {ds} -o {base_pre}/p2rank -threads 1"

#执行命令
os.system(cmd)

# 创建空列表，用于存储信息
info = []

# 获取分子的smiles信息
for i, line in tqdm(m.iterrows(), total=m.shape[0]):
    smiles = line['SMILES']

    # 设置化合物名称为空字符串
    compound_name = ""

    # 设置蛋白质名称
    protein_name = proteinName

    #     # use protein center as the pocket center.
    #     # 计算蛋白质中心坐标
    #     com = ",".join([str(a.round(3)) for a in protein_dict[proteinName][0].mean(axis=0).numpy()])

    #     # 将蛋白质名称、化合物名称、smiles、口袋名称和口袋中心坐标添加到info列表
    #     info.append([protein_name, compound_name, smiles, "protein_center", com])
    #     # 由于WDR实际上足够小，我们感兴趣的是找到一个与中心空腔结合的配体。
    #     # 以蛋白质质心为中心的区块就足够了。
    #     # 我们不需要额外的p2rank预测中心。
    #     if False:

    # 设置p2rank预测文件路径
    p2rankFile = f"{base_pre}/p2rank/{proteinName}.pdb_predictions.csv"

    # 读取p2rank预测文件
    pocket = pd.read_csv(p2rankFile)

    # 清除列名的空白字符
    pocket.columns = pocket.columns.str.strip()

    # 获取排名第一的口袋
    top_pocket = pocket.iloc[0]

    # 获取口袋中心坐标
    top_pocket_com = top_pocket[['center_x', 'center_y', 'center_z']].values

    # 将中心坐标转换为字符串格式
    top_pocket_com_str = ",".join([str(a.round(3)) for a in top_pocket_com])

    # 将蛋白质名称、化合物名称、smiles、口袋名称和口袋中心坐标添加到info列表
    info.append([protein_name, compound_name, smiles, "top_pocket", top_pocket_com_str])


# 将info列表转换为pandas DataFrame并设置列名
info = pd.DataFrame(info, columns=['protein_name', 'compound_name', 'smiles', 'pocket_name', 'pocket_com'])

info.to_csv('/home/dell/wangzhen/TankBind-main/examples/HTVS/info/info.csv', index=False)

