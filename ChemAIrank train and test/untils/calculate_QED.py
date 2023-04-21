import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import QED
""""""
# # 设置文件夹路径
# data_dir = "/home/dell/wangzhen/RealQED(2.17)/test/data/finetune/toxcast"

# # 获取所有文件夹的路径列表
# sub_dirs = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]

# for sub_dir in sub_dirs:
#     # 获取所有csv文件的路径列表
#     csv_files = [os.path.join(sub_dir, f) for f in os.listdir(sub_dir) if f.endswith('.csv') and 'prediction_new' in f]

# for csv_file in csv_files:
csv_file = '/home/dell/wangzhen/RealQED(2.17)/test/data/anticancer/anticancer_smiles206_prediction_new.csv'
# 读取CSV文件
df = pd.read_csv(csv_file)
# 重命名Prediction列为ChemAIra
df = df.rename(columns={"Prediction": "ChemAIrank"})
# 计算分子的QED值
qed_values = []
for smiles in df['smiles']:
    mol = Chem.MolFromSmiles(smiles)
    qed_value = QED.qed(mol) * 100
    qed_values.append(round(qed_value, 2))
# 将QED值覆盖原来的QED列
df['QED'] = qed_values
# 保留小数点后两位
df['ChemAIrank'] = df['ChemAIrank'].round(2)
df['QED'] = df['QED'].round(2)
# 保存结果
df.to_csv(csv_file, index=False)
