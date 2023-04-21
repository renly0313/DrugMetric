import pandas as pd
from rdkit import Chem
from rdkit.Chem import QED
import numpy as np
# 定义数据集标签
labels = ['Candidate drug', 'ChEMBL', 'ZINC', 'GDB']

# 读取数据集文件
df_candidate = pd.read_csv('/home/dell/wangzhen/RealQED(2.17)/test/data/four_dataset_test/Candidate_drug_test.txt', header=None, names=['smiles'])
df_chembl = pd.read_csv('/home/dell/wangzhen/RealQED(2.17)/test/data/four_dataset_test/ChEMBL_seed0_test.txt', header=None, names=['smiles'])
df_zinc = pd.read_csv('/home/dell/wangzhen/RealQED(2.17)/test/data/four_dataset_test/zinc_rand1_test.txt', header=None, names=['smiles'])
df_gdb = pd.read_csv('/home/dell/wangzhen/RealQED(2.17)/test/data/four_dataset_test/gdb_rand1_test.txt', header=None, names=['smiles'])

# 计算QED值
def calculate_qed(smiles):
    mol = Chem.MolFromSmiles(smiles)
    qed_value = np.round(QED.qed(mol)*100,2)
    return qed_value

# 添加QED列和标签列
for df, label in zip([df_candidate, df_chembl, df_zinc, df_gdb], labels):
    df['qed'] = df['smiles'].apply(calculate_qed)
    df['datasets'] = label

# 合并数据集
df_all = pd.concat([df_candidate, df_chembl, df_zinc, df_gdb], ignore_index=True)

# 保存为新文件
df_all.to_csv('/home/dell/wangzhen/RealQED(2.17)/test/result/perdiciton/all_datasets_test_rand1_qed.csv', index=False)
