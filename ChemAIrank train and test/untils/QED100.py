import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import QED
import numpy as np
# 读取CSV文件
df = pd.read_csv("../test/data/LARGE/affinity_large_than_8_filtered.csv")
# 计算分子的QED值
qed_values = []
for smiles in df['smiles']:
    mol = Chem.MolFromSmiles(smiles)
    qed_value = np.round(QED.qed(mol)*100, 2)
    qed_values.append(qed_value)
# 将QED值添加为新的一列
df['QED'] = qed_values
# 保存结果
df.to_csv("../test/data/LARGE/affinity_large_than_8_filtered.csv", index=False)
