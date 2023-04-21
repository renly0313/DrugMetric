import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# 从csv文件中读取数据集
clinical_drug = pd.read_csv('../../data/datasets/new_data/all_sample_train_data/all_test_data_pro_datasets.csv')

# 从数据集中筛选clinical_drug数据集，并给该数据集中所有数据标记为正样本
p = clinical_drug[clinical_drug['datasets'].isin(['clinical_drug'])]
p.insert(p.shape[1], 'label', 1)

# 从数据集中筛选zinc数据集，并给该数据集中所有数据标记为负样本
n = clinical_drug[clinical_drug['datasets'].isin(['zinc'])]
n.insert(n.shape[1], 'label', 0)

# 合并正样本数据集和负样本数据集
data = pd.concat([p, n])

# 定义计算auroc的函数，函数的输入是正样本集和负样本集
def compute_auroc(pos, neg):
    # 将正样本和负样本的真实标签存入true_list中，正样本为1，负样本为0
    true_list = np.array([1 for _ in range(len(pos))] + [0 for _ in range(len(neg))])
    # 将正样本和负样本的预测得分存入score_list中
    score_list = np.array(pos + neg)
    # 计算auroc并返回结果
    return roc_auc_score(true_list, score_list)

# 将正样本的qed值和负样本的qed值分别存入pos_set和neg_set中
pos_set = p['qed'].tolist()
neg_set = n['qed'].tolist()

# 调用compute_auroc函数计算auroc
auroc = compute_auroc(pos_set, neg_set)

# 输出auroc的值
print(auroc)

# 计算fpr和tpr并绘制ROC曲线
fpr, tpr, thresholds = roc_curve(np.array([1 for _ in range(len(pos_set))] + [0 for _ in range(len(neg_set))]), np.array(pos_set + neg_set))
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], '--')
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
