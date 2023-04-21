import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

# 从输入中读取目标分子的SMILES
target_smiles = "FC(c1cc2cnc(nc2n(c1=O)C1CCCC1(C)O)NC1CCN(CC1)S(=O)(=O)C)F"

# 从CSV文件中读取SMILES列表
csv_file_path = "/home/dell/wangzhen/TankBind-main/examples/HTVS/simlarity_compare/affinity_large_than_8_filtered.csv"
df = pd.read_csv(csv_file_path)
smiles_list = df["SMILES"].tolist()

# 计算目标分子与每个分子的谷本相似度
def compute_tanimoto_similarity(smiles1, smiles2):
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2)
    return DataStructs.TanimotoSimilarity(fp1, fp2)

# 初始化一个空的字典，用于存储谷本相似度
tanimoto_similarity_dict = {}

# 计算相似度
for smiles in smiles_list:
    tanimoto_similarity = compute_tanimoto_similarity(target_smiles, smiles)
    tanimoto_similarity_dict[smiles] = tanimoto_similarity

# 输出谷本相似度
for smiles, similarity in tanimoto_similarity_dict.items():
    print(f"SMILES: {smiles}, Tanimoto similarity: {similarity}")

# 添加谷本相似度到数据框
df['Tanimoto_similarity'] = df['SMILES'].map(tanimoto_similarity_dict)

# 保存数据框到CSV文件
output_csv_path = "/home/dell/wangzhen/TankBind-main/examples/HTVS/simlarity_compare/affinity_large_than_8_filtered_with_similarity.csv"
df.to_csv(output_csv_path, index=False)
