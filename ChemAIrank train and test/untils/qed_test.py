import csv
from rdkit import Chem
from rdkit.Chem import QED

# 指定CSV文件路径和分子SMILES所在列的索引
csv_file = '/home/dell/wangzhen/RealQED(2.17)/test/data/admetlab2/qm7_prediction_new.csv'
smiles_column_index = 0

# 读取CSV文件并获取分子SMILES列表
smiles_list = []
with open(csv_file, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        smiles_list.append(row[smiles_column_index])

# 计算分子的QED并计算平均值
qed_list = []
for smiles in smiles_list:
    mol = Chem.MolFromSmiles(smiles)
    qed = QED.qed(mol)
    qed_list.append(qed)

average_qed = sum(qed_list) / len(qed_list)
print(f"所有分子的平均QED值为：{average_qed:.2f}")
