import os
import pandas as pd

# 读取CSV文件
data = pd.read_csv(
    '/home/dell/wangzhen/TankBind-main/examples/HTVS/simlarity_compare/affinity_large_than_8_filtered.csv')

# 按ChemAIrank排序
data_sorted = data.sort_values(by='ChemAIrank', ascending=False)

# 计算每个区间的数量
total_count = len(data_sorted)
interval = total_count // 10

# 创建10个子集
subsets = []
for i in range(10):
    start_index = i * 100
    end_index = (i + 1) * 100
    subset = data_sorted.iloc[start_index:end_index]
    subsets.append(subset)

# 保存到10个文件夹
for i, subset in enumerate(subsets):
    folder_path = f'./output/folder_{i + 1}'
    file_name = f'{folder_path}/subset_{i + 1}.csv'

    # 如果文件夹不存在，则创建
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # 保存子集为CSV文件
    subset.to_csv(file_name, index=False)

    # 提取SMILES列并保存到新文件
    smiles_file_name = f'{folder_path}/smiles_{i + 1}.txt'
    subset['SMILES'].to_csv(smiles_file_name, index=False, header=False)

    print("分子已按比例保存到10个文件夹中。")
