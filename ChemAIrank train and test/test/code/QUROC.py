import os
import argparse
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import QED
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# 构建命令行参数解析器
parser = argparse.ArgumentParser(description='Calculate and plot ROC curve from molecular SMILES in txt files')
parser.add_argument('--data_dir', type=str, default='./data', help='Directory path for txt files containing molecular SMILES')
parser.add_argument('--pos_file', type=str, default='pos_smiles.txt', help='Txt file containing positive molecular SMILES')
parser.add_argument('--neg_file', type=str, default='neg_smiles.txt', help='Txt file containing negative molecular SMILES')
args = parser.parse_args()

# 拼接文件路径
pos_path = os.path.join(args.data_dir, args.pos_file)
neg_path = os.path.join(args.data_dir, args.neg_file)

# 从txt文件中读取分子smiles
with open(pos_path, 'r') as f:
    pos_smiles = f.readlines()
with open(neg_path, 'r') as f:
    neg_smiles = f.readlines()

# 去除分子smiles中的换行符
pos_smiles = [s.strip() for s in pos_smiles]
neg_smiles = [s.strip() for s in neg_smiles]

# 计算分子的qed值
pos_qed = [QED.qed(Chem.MolFromSmiles(s)) for s in pos_smiles]
neg_qed = [QED.qed(Chem.MolFromSmiles(s)) for s in neg_smiles]

# 计算auroc
true_list = np.array([1 for _ in range(len(pos_qed))] + [0 for _ in range(len(neg_qed))])
score_list = np.array(pos_qed + neg_qed)
auroc = roc_auc_score(true_list, score_list)

# 输出auroc的值
print(auroc)

# 计算fpr和tpr并绘制ROC曲线
fpr, tpr, thresholds = roc_curve(true_list, score_list)
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], '--')
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
