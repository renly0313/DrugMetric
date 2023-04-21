import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import os
# 读取CSV文件
data_path = '/home/dell/wangzhen/RealQED(2.17)/test/data/admetlab2/result/4databset_test_admet.csv'
data = pd.read_csv(data_path)

# 移除包含缺失值的行
data = data.dropna()

# 按QED和ChemAIrank排序
data_qed = data.sort_values(by='QED', ascending=False)
data_chemai = data.sort_values(by='ChemAIrank', ascending=False)

# 提取高低25分子
top_25_qed = data_qed.head(25)
bottom_25_qed = data_qed.tail(25)
top_25_chemai = data_chemai.head(25)
bottom_25_chemai = data_chemai.tail(25)
# 计算p值的函数
def calculate_pvalues(data1, data2):
    data1 = data1.apply(pd.to_numeric, errors='coerce')
    data2 = data2.apply(pd.to_numeric, errors='coerce')
    data1 = data1.dropna(axis=1)
    data2 = data2.dropna(axis=1)
    features = data1.columns.drop(['QED', 'ChemAIrank'])
    pvalues = {}
    for feature in features:
        _, pvalue = ttest_ind(data1[feature], data2[feature])
        pvalues[feature] = pvalue
    return pvalues

# 计算QED排序下的p值
pvalues_qed = calculate_pvalues(top_25_qed, bottom_25_qed)

# 计算ChemAIrank排序下的p值
pvalues_chemai = calculate_pvalues(top_25_chemai, bottom_25_chemai)

def calculate_significant_features(pvalues):
    significant_features = {
        '<0.001': [],
        '<0.01': [],
        '<0.05': []
    }
    for feature, pvalue in pvalues.items():
        if pvalue < 0.001:
            significant_features['<0.001'].append(feature)
        elif pvalue < 0.01:
            significant_features['<0.01'].append(feature)
        elif pvalue < 0.05:
            significant_features['<0.05'].append(feature)
    return significant_features

def print_significant_features(significant_features):
    print('具有显著差异的特征：')
    for level, features in significant_features.items():
        print(f'p值{level}的特征：{", ".join(features)}')

def add_pvalue(ax, pvalue):
    text = ''
    if pvalue < 0.001:
        text = f'P-值<0.001***'
    elif pvalue < 0.01:
        text = f'P-值<0.01**'
    elif pvalue < 0.05:
        text = f'P-值<0.05*'
    else:
        text = f'P-值：{pvalue}'
    ax.annotate(text, xy=(0.6, 0.9), xycoords='axes fraction', fontsize=10)

# 绘制箱线图的函数
def plot_boxplots(data1, data2, title, pvalues, significant_features):
    n = len(significant_features['<0.001']) + len(significant_features['<0.01']) + len(significant_features['<0.05'])
    ncols = 3
    nrows = 2
    ngroups = (n + ncols * nrows - 1) // (ncols * nrows)
    for group_idx in range(ngroups):
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 10))
        idx = group_idx * ncols * nrows
        for level, features in significant_features.items():
            for feature in features:
                if idx >= (group_idx + 1) * ncols * nrows:
                    break
                row = idx % (ncols * nrows) // ncols
                col = idx % ncols
                ax = axes[row, col]
                ax.boxplot([data1[feature], data2[feature]], labels=['前25个', '后25个'])
                ax.set_title(f'{feature} - {title}')
                add_pvalue(ax, pvalues[feature])
                idx += 1
        plt.tight_layout()
        plt.show()

# 计算QED排序下的显著特征
significant_features_qed = calculate_significant_features(pvalues_qed)
print('QED排序：')
print_significant_features(significant_features_qed)

# 计算ChemAIrank排序下的显著特征
significant_features_chemai = calculate_significant_features(pvalues_chemai)
print('\nChemAIrank排序：')
print_significant_features(significant_features_chemai)

# 绘制QED排序下的箱线图
plot_boxplots(top_25_qed, bottom_25_qed, 'QED', pvalues_qed, significant_features_qed)

# 绘制ChemAIrank排序下的箱线图
plot_boxplots(top_25_chemai, bottom_25_chemai, 'ChemAIrank', pvalues_chemai, significant_features_chemai)
