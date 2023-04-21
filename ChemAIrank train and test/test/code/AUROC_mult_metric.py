import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# 从csv文件中读取数据集
Candidate_drug = pd.read_csv('/home/dell/wangzhen/RealQED(2.17)/test/data/admetlab2/result/4databset_test_admet.csv')
color_map = {'QED': '#9467bd', 'ChemAIrank': '#1f77b4', 'Lipinski': '#d62728',
             'Pfizer': '#2ca02c', 'GSK': '#ff7f0e', 'GoldenTriangle': '#8c564b'}
# 从数据集中筛选candidate_drug数据集，并给该数据集中所有数据标记为正样本
p = Candidate_drug[Candidate_drug['datasets'].isin(['Candidate_drug'])]
p.insert(p.shape[1], 'label', 1)

# 从数据集中筛选zinc数据集，并给该数据集中所有数据标记为负样本
n = Candidate_drug[Candidate_drug['datasets'].isin(['GDB'])]
n.insert(n.shape[1], 'label', 0)

# 合并正样本数据集和负样本数据集
data = pd.concat([p, n])

# 定义计算auroc的函数，函数的输入是正样本集和负样本集
def compute_auroc(pos, neg):
    true_list = np.array([1 for _ in range(len(pos))] + [0 for _ in range(len(neg))])
    score_list = np.array(pos + neg)
    return roc_auc_score(true_list, score_list)

# 定义特征列表
features = ['QED', 'ChemAIrank', 'Lipinski', 'Pfizer', 'GSK', 'GoldenTriangle']

# 定义正样本集和负样本集字典
pos_sets = {}
neg_sets = {}

# 定义auroc字典
aurocs = {}

# 计算每个特征的正样本集、负样本集和auroc
# 用.loc方法替换原来的赋值操作
for feature in features:
    p.loc[:, feature] = p[feature].apply(lambda x: 1 if x == 'Accepted' else (0 if x == 'Rejected' else x))
    n.loc[:, feature] = n[feature].apply(lambda x: 1 if x == 'Accepted' else (0 if x == 'Rejected' else x))
    pos_sets[feature] = p[feature].tolist()
    neg_sets[feature] = n[feature].tolist()
    aurocs[feature] = compute_auroc(pos_sets[feature], neg_sets[feature])

# 输出各特征的auroc值
for feature, auroc in aurocs.items():
    print('{} auroc: {:.2f}'.format(feature, auroc))

# 根据auroc值对特征排序
sorted_features = sorted(features, key=lambda x: aurocs[x], reverse=True)

# 按照排序后的特征顺序绘制ROC曲线
for feature in sorted_features:
    fpr, tpr, _ = roc_curve(np.array([1 for _ in range(len(pos_sets[feature]))] + [0 for _ in range(len(neg_sets[feature]))]), np.array(pos_sets[feature] + neg_sets[feature]))
    plt.plot(fpr, tpr, label='{} ({:.2f})'.format(feature, aurocs[feature]), color=color_map[feature])  # 使用指定颜色绘制ROC曲线
# 绘制对角线
plt.plot([0, 1], [0, 1], '--')

# 设置图表标题、坐标轴标签和图例
dataset_name = data['datasets'].unique()
dataset_name_str = '_'.join(dataset_name)
title = ' {} ROC Curve'.format('/'.join(dataset_name))

# plt.title(title)
plt.plot([0, 1], [0, 1], '--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right',  fontsize=8)  # 修改这里，将图例放在右下角
# 保存图像到指定路径，设置分辨率为300 dpi
output_path = "/home/dell/wangzhen/RealQED(2.17)/test/result/AUROC/{}_ROC_curve.png".format(dataset_name_str)
plt.savefig(output_path, dpi=300)
plt.show()

