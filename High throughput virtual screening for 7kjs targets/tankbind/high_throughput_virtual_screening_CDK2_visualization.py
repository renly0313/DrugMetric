import matplotlib.pyplot as plt
import pandas as pd
# Load the prediction results CSV file
info = pd.read_csv('/home/dell/wangzhen/TankBind-main/examples/HTVS/info/info.csv')

from generation_utils import get_LAS_distance_constraint_mask, get_info_pred_distance, write_with_new_coords
# 选择亲和力大于7的项
info = info.query("affinity > 8").reset_index(drop=True)

# #
# # 按亲和力降序排序
# info = info.sort_values(by='affinity', ascending=False)
# # 选择亲和力排名前1000的项
# info = info.head(1000).reset_index(drop=True)

# 获取第一条数据
line = chosen.iloc[0]
idx = line['index']
one_data = dataset[idx]

# 使用数据加载器处理数据
data_with_batch_info = next(iter(DataLoader(dataset[idx:idx+1], batch_size=1,
                         follow_batch=['x', 'y', 'compound_pair'], shuffle=False, num_workers=1)))
data_with_batch_info = data_with_batch_info.to(device)  # 将数据转移到 GPU
y_pred, affinity_pred = model(data_with_batch_info)

# 获取坐标
coords = one_data.coords.to(device)
protein_nodes_xyz = one_data.node_xyz.to(device)
n_compound = coords.shape[0]
n_protein = protein_nodes_xyz.shape[0]
y_pred = y_pred.reshape(n_protein, n_compound).to(device).detach()
y = one_data.dis_map.reshape(n_protein, n_compound).to(device)

# 计算化合物距离约束
compound_pair_dis_constraint = torch.cdist(coords, coords)
