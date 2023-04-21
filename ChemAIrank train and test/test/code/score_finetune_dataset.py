import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib.cm as cm

data_path = "/home/dell/wangzhen/RealQED(2.17)/test/data/finetune"
datasets = {}

# 循环读取每个文件夹下的csv文件
for folder in os.listdir(data_path):
    folder_path = os.path.join(data_path, folder)
    if os.path.isdir(folder_path):
        csv_files = [f for f in os.listdir(folder_path) if f.endswith('prediction_new.csv')]
        for csv_file in csv_files:
            dataset_name = f"{folder}_{csv_file.split('_')[0]}"
            # 判断是否为lipo或bace数据集，如果是则跳过该数据集
            if 'lipo' in dataset_name or 'bace' in dataset_name:
                continue
            file_path = os.path.join(folder_path, csv_file)
            df = pd.read_csv(file_path, usecols=['smiles', 'ChemAIrank', 'QED'])
            datasets[dataset_name] = df

# 取出ChemAIrank列并绘制小提琴图
ChemAIrank_list = []
for dataset_name, dataset_df in datasets.items():
    ChemAIrank_list.append(dataset_df['ChemAIrank'].tolist())
num_datasets = len(datasets)
stats_length = len(ChemAIrank_list)
positions = list(range(num_datasets))[:stats_length] # 调整 positions 的长度
# 手动创建颜色列表
# colors = ['#ff0000', '#ff3300', '#ff6600', '#ff9900', '#ffcc00', '#ffff00', '#ccff00', '#99ff00', '#66ff00', '#33ff00', '#00ff00']# 绘制小提琴图
# colors = ['#ff0000', '#ff3300', '#ff6600', '#ff9900',
#           '#ffff00', '#ccff00', '#66ff00', '#33ff00', '#00ff00']# 绘制小提琴图
colors = ['#00ff00','#33ff00', '#66ff00', '#ccff00', '#ffff00', '#ff9900', '#ff6600', '#ff3300', '#ff0000']# 绘制小提琴图
# 计算均值、极值、方差
mean_values = [np.mean(x) for x in ChemAIrank_list]
max_values = [np.max(x) for x in ChemAIrank_list]
min_values = [np.min(x) for x in ChemAIrank_list]



# 输出均值、极值
for i, dataset_name in enumerate(datasets.keys()):
    print(f"{dataset_name.split('_')[0]}:\n"
          f"\tMean: {mean_values[i]:.2f}\n"
          f"\tMax: {max_values[i]}\n"
          f"\tMin: {min_values[i]}\n")


# 按均值排序
sorted_indexes = sorted(range(len(mean_values)), key=lambda i: mean_values[i])
sorted_ChemAIrank_list = [ChemAIrank_list[i] for i in sorted_indexes]
sorted_mean_values = [mean_values[i] for i in sorted_indexes]
sorted_max_values = [max_values[i] for i in sorted_indexes]
sorted_min_values = [min_values[i] for i in sorted_indexes]




# 绘制小提琴图，按均值排序
fig, ax = plt.subplots()
ax.set_xlim(0, 100)
parts = ax.violinplot(sorted_ChemAIrank_list, showmeans=True, showextrema=True, positions=positions, widths=0.8,
                      bw_method='silverman', vert=False, points=800)
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors[i % len(colors)])
parts['cmeans'].set_color('#3f3f3f')
parts['cmaxes'].set_color('#3f3f3f')
parts['cmins'].set_color('#3f3f3f')
parts['cbars'].set_color('#3f3f3f')
ax.set_yticks(range(0, num_datasets))
ax.set_yticklabels([list(datasets.keys())[i].split("_")[0] for i in sorted_indexes], rotation=0, ha='right')
ax.set_xlabel('ChemAIrank')
ax.set_ylabel('Datasets')
# ax.set_title('ChemAIrank Violin Plot ')
plt.savefig('/home/dell/wangzhen/RealQED(2.17)/test/result/violin_plot_sorted_ChemAIrank9.png', dpi=300, bbox_inches='tight')
plt.show()
