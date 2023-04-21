import pandas as pd

# 读取数据并自动识别分隔符，不将任何列作为索引
data = pd.read_csv('/home/dell/wangzhen/TankBind-main/examples/HTVS/result/10.csv', sep='\s+|,|\t', engine='python', index_col=None)

# 添加索引
data = data.reset_index()

# 截取到 CDK2_pki 列（包含该列）
columns_to_keep = data.columns[:data.columns.get_loc('CDK2_pki') + 1]
data = data[columns_to_keep]

# 将列名整体向后移动一格
old_columns = data.columns[:-1]
new_columns = data.columns[1:]
col_mapping = dict(zip(old_columns, new_columns))
data = data.rename(columns=col_mapping)

# 删除多余的第一列
data = data.iloc[:, 1:]
data = data.iloc[:, :-1]
data.to_csv('/home/dell/wangzhen/TankBind-main/examples/HTVS/result/10_.csv', index=False)
