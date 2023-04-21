import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# 读取数据
train = pd.read_csv('/home/dell/wangzhen/RealQED(2.17)/test/result/perdiciton/all_datasets_test_rand1_prediction_scores.csv')

# 设置颜色
colors = ["#8c009c", "#c80457", "#dd7608", "#0034b4"]

# 对数据进行排序，将GDB放在最后
train['datasets'] = pd.Categorical(train['datasets'], categories=['Candidate drug', 'ChEMBL', 'ZINC', 'GDB'], ordered=True)
train.sort_values('datasets', inplace=True)

# 分组数据
grouped = train.groupby("datasets")["ChemAIrank"]

# 准备数据
data = [grouped.get_group(g).values for g in grouped.groups]

# 创建小提琴图
fig, ax = plt.subplots()
parts = ax.violinplot(data, showmeans=True, showextrema=True, widths=0.8, bw_method='silverman', points=800)

# 设置颜色
for pc, color in zip(parts['bodies'], colors):
    pc.set_facecolor(color)
    pc.set_edgecolor('black')

# 设置极值和轴颜色
parts['cmeans'].set_color('#3f3f3f')
parts['cmaxes'].set_color('#3f3f3f')
parts['cmins'].set_color('#3f3f3f')
parts['cbars'].set_color('#3f3f3f')


# 设置X轴标签
ax.set_xticks(range(1, len(grouped.groups) + 1))
ax.set_xticklabels(grouped.groups.keys())

# 设置Y轴范围
ax.set_ylim(0, 100)

# 设置图的标题
# plt.title("ChemAIra Distribution")

# 保存图片
plt.savefig("/home/dell/wangzhen/RealQED(2.17)/test/result/violin_plot_ChemAIrank", dpi=300)

# 显示图片
plt.show()

# Calculate and print the mean values for each dataset
means = grouped.mean()
print("Mean values for each dataset:")
print(means)
