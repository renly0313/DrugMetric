import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import os

# 读取CSV文件
data_path = '/home/dell/wangzhen/RealQED(2.17)/test/data/admetlab2/result/candidate_drug_admet.csv'
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

# 定义特征全称
features_fullname = {
    'Vdss': 'volume of distribution',
    'fu': 'fraction unbound in plasma',
    'CLt': 'total clearance',
    'CLh': 'hepatic clearance',
    'CLr': 'renal clearance',
    'F': 'Bioavailability',
    'Fa': 'Fraction absorbed',
    'Fg': 'fraction escaping gut-wall elimination',
    'Fh': 'fraction escaping first-pass hepatic elimination'
}

# 计算pvalue的函数
def calculate_pvalues(data1, data2, features):
    pvalues = {}
    for feature in features:
        _, pvalue = ttest_ind(data1[feature], data2[feature])
        pvalues[feature] = round(pvalue, 2)
    return pvalues

# 在子图上添加pvalue
def add_pvalue(ax, pvalue):
    text = ''
    if pvalue < 0.001:
        text = f'P-Value<0.001***'
    elif pvalue < 0.01:
        text = f'P-Value<0.01**'
    elif pvalue < 0.05:
        text = f'P-Value<0.05*'
    else:
        text = f'P-Value：{pvalue}'
    ax.annotate(text, xy=(0.6, 0.9), xycoords='axes fraction', fontsize=10)

# 修改绘制箱线图的函数
def plot_boxplots(data1, data2, title, pvalues, features):
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 12), dpi=300)
    for i, feature in enumerate(features):
        row = i // 3
        col = i % 3
        bplot = axes[row, col].boxplot([data1[feature], data2[feature]], labels=['Top 25', 'Bottom 25'],
                                        patch_artist=True, widths=0.5)
        # 设置颜色
        colors = ['#e1812c', '#3274a1']
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        axes[row, col].set_title(f'{features[feature]} - {title}')
        add_pvalue(axes[row, col], pvalues[feature])
    plt.tight_layout()
    save_path = f'/home/dell/wangzhen/RealQED(2.17)/test/result/box_plot/{title}_box_plot.png'
    plt.savefig(save_path, dpi=300)
    plt.show()

# 创建结果文件夹
result_dir = '/home/dell/wangzhen/RealQED(2.17)/test/result/box_plot'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# 计算QED排序的pvalue
pvalues_qed = calculate_pvalues(top_25_qed, bottom_25_qed, features_fullname)

# 绘制QED排序的箱线图
plot_boxplots(top_25_qed, bottom_25_qed, 'QED', pvalues_qed, features_fullname)

# 计算ChemAIrank排序的pvalue
pvalues_chemai = calculate_pvalues(top_25_chemai, bottom_25_chemai, features_fullname)

# 绘制ChemAIrank排序的箱线图
plot_boxplots(top_25_chemai, bottom_25_chemai, 'ChemAIrank', pvalues_chemai, features_fullname)
